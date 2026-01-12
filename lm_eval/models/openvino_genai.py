import logging
import numpy as np
from typing import List, Optional
import copy
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("openvino_genai")
class OpenVINOGenAILM(HFLM):
    """
    OpenVINO GenAI backend for lm-evaluation-harness.
    Uses openvino_genai library.
    
    This provides native integration with OpenVINO GenAI for accelerated inference
    on Intel architectures (CPU, GPU, NPU).
    """

    def __init__(
        self,
        device="cpu",
        config=None,
        **kwargs,
    ) -> None:
        # Define config as a class attribute first
        self._config = config if config is not None else {}
        
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only causal models are supported."
            )

        self.openvino_device = device.upper() if isinstance(device, str) else "CPU"
        
        # Initialize pipeline and tokenizer as None - will be created in _create_model
        self._pipeline = None
        self._tokenizer = None

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        gpus=0,
        offload_folder='./offload',
        autogptq=False,
        gptqmodel=False,
        parallelize=False,
        **kwargs,
    ) -> None:
        """Create OpenVINO GenAI pipeline"""
        try:
            import openvino_genai
        except ImportError:
            raise ModuleNotFoundError(
                "package `openvino_genai` is not installed. "
                "Please install it via `pip install openvino-genai`"
            )

        # Properties for LLMPipeline
        # disable_slice_optimization is REQUIRED for echo mode to work correctly
        # Slice optimization transforms the model graph to compute only last token logits,
        # which breaks echo mode that needs correct log_probs for all prompt positions
        pipeline_properties = {
            "disable_slice_optimization": True,
        }
        
        # Create OpenVINO GenAI pipeline
        eval_logger.info(f"Creating OpenVINO GenAI pipeline: model={pretrained}, device={self.openvino_device}")
        self._pipeline = openvino_genai.LLMPipeline(
            models_path=pretrained,
            device=self.openvino_device,
            **pipeline_properties,
        )
        
        # Get tokenizer from pipeline
        self._tokenizer = self._pipeline.get_tokenizer()
        eval_logger.info("OpenVINO GenAI pipeline created successfully")
    
    @property
    def ov_tokenizer(self):
        """Access the OpenVINO tokenizer"""
        return self._tokenizer
    
    @property
    def model(self):
        """Return pipeline as model for HFLM compatibility"""
        return self._pipeline
    
    @property
    def _model(self):
        """Alias for compatibility with parent class"""
        return self._pipeline
    
    def loglikelihood(self, requests):
        """
        Calculate loglikelihood using OpenVINO GenAI with echo=True to get prompt logprobs.
        Direct implementation using openvino_genai.LLMPipeline.
        """
        try:
            import openvino_genai
            GenerationConfig = openvino_genai.GenerationConfig
        except ImportError:
            raise ModuleNotFoundError(
                "package `openvino_genai` is not installed. Please install it via `pip install openvino-genai`"
            )
        
        res = []
        for request in requests:
            context, continuation = request.args
            
            # Tokenize full text
            whole_enc = self._tokenizer.encode(context + continuation)
            
            # Configure for echo mode
            generation_config = GenerationConfig()
            generation_config.echo = True
            generation_config.max_new_tokens = 0  # Don't generate, just compute logprobs
            generation_config.do_sample = False
            
            try:
                # Generate with echo mode - direct pipeline call
                result = self._pipeline.generate(whole_enc, generation_config)
                log_probs = result.log_probs[0] if result.log_probs else []
                
                if len(log_probs) == 0:
                    eval_logger.warning("Received empty log_probs, returning fake value")
                    fake_loglikelihood = -1.0
                    fake_is_greedy = True
                    res.append((fake_loglikelihood, fake_is_greedy))
                    continue
                
                # Find continuation start position
                context_enc = self._tokenizer.encode(context)
                context_tokens = context_enc.input_ids.data[0] if len(context_enc.input_ids.data) > 0 else []
                context_len = len(context_tokens)
                
                eval_logger.debug(f"context_len={context_len}, log_probs_len={len(log_probs)}")
                
                # Extract continuation log_probs
                # After fix in openvino.genai: log_probs[i] is the log probability of token i (0-indexed)
                # So continuation tokens start at context_len
                continuation_log_probs = log_probs[context_len:]
                
                # Sum log probabilities for the continuation
                answer = sum(continuation_log_probs)
                
                # For greedy decoding (do_sample=False)
                is_greedy = True
                
                res.append((answer, is_greedy))
                
            except Exception as e:
                eval_logger.error(f"Error computing loglikelihood: {e}")
                # Return fake value on error
                res.append((-1.0, True))
        
        return res

    def loglikelihood_rolling(self, requests):
        """
        Return fake rolling loglikelihood values for evaluation purposes.
        OpenVINO GenAI models are focused on generation and don't support loglikelihood calculation.
        """
        eval_logger.warning(
            "OpenVINO GenAI models don't support loglikelihood calculation. Returning fake values."
        )
        
        res = []
        for request in requests:
            context, continuation = request
            # Return fake token loglikelihoods - one value per token in continuation
            fake_token_loglikelihoods = [-1.0] * len(continuation)  # Fake values
            res.append(fake_token_loglikelihoods)
        
        return res 

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text using OpenVINO GenAI's pipeline until a specified stopping criteria is met.
        Maintains compatibility with the HFLM implementation.
        
        Args:
            requests: List of Instance objects containing generation requests
            disable_tqdm: Whether to disable the progress bar
        
        Returns:
            List of generated strings
        """
        res = []
        
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running OpenVINO GenAI generation requests",
        )
        
        # Process each request individually
        for request in requests:
            context, gen_kwargs = request.args
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                stop_strings = kwargs.pop("until", None)
                # Extract max_gen_toks if provided, otherwise use default
                if "max_gen_toks" in kwargs.keys() and "max_new_tokens" in kwargs.keys():
                    logging.warning("Set max_gen_toks and max_new_tokens in meantime, will set with max_new_tokens")
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)

                if "max_new_tokens" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_new_tokens")
            else:
                raise ValueError(f"Expected kwargs to be of type dict but got {type(gen_kwargs)}")
            
            try:
                # Set up generation parameters
                generation_kwargs = {
                    "max_new_tokens": max_gen_toks,
                    "stop_strings": set(stop_strings),
                }
                
                # Add any other parameters from kwargs that are supported by OpenVINO GenAI
                for k, v in kwargs.items():
                    if k not in generation_kwargs:
                        generation_kwargs[k] = v

                generated_text = self._model.generate(
                    context,
                    **generation_kwargs
                )
                    
                res.append(generated_text)
                
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
                
            except Exception as e:
                eval_logger.error(f"Error during generation: {e}")
                res.append("")
            
            pbar.update(1)
        
        pbar.close()
        return res 
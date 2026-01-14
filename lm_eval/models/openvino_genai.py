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
        
        Optimized for multiple-choice tasks (like MMLU):
        - Groups requests by context
        - For single-token continuations sharing the same context: processes them together
        """
        try:
            import openvino_genai
            GenerationConfig = openvino_genai.GenerationConfig
        except ImportError:
            raise ModuleNotFoundError(
                "package `openvino_genai` is not installed. Please install it via `pip install openvino-genai`"
            )
        
        # Group requests by context for optimization
        grouped_requests = {}
        for idx, request in enumerate(requests):
            context, continuation = request.args
            if context not in grouped_requests:
                grouped_requests[context] = []
            grouped_requests[context].append((idx, continuation))
        
        # Results array indexed by original request position
        res = [None] * len(requests)
        
        # Configure for echo mode
        generation_config = GenerationConfig()
        generation_config.echo = True
        generation_config.max_new_tokens = 0  # Don't generate, just compute logprobs
        generation_config.do_sample = False
        
        # Create progress bar
        pbar = tqdm(
            total=len(grouped_requests),
            disable=(self.rank != 0),
            desc="Processing requests"
        )
        
        for context, continuations in grouped_requests.items():
            # Tokenize context once
            context_enc = self._tokenizer.encode(context)
            context_tokens = context_enc.input_ids.data[0] if len(context_enc.input_ids.data) > 0 else []
            context_len = len(context_tokens)
            
            # Group continuations by type for optimization
            single_token_continuations = []
            multi_token_continuations = []
            
            # Get BOS token ID to strip it from continuations
            # Different models use different BOS tokens:
            # - Llama 3.x: 128000, Llama 2.x: 1, Mistral: 1, etc.
            bos_token_id = getattr(self._tokenizer, 'bos_token_id', None)
            
            # If not available, try to detect it from a sample encoding
            if bos_token_id is None and continuations:
                # Encode a simple string and check if first token is a special token
                # Common BOS patterns: very high ID (128000+) or ID=1
                sample_enc = self._tokenizer.encode(" test")
                sample_tokens = list(sample_enc.input_ids.data[0]) if len(sample_enc.input_ids.data) > 0 else []
                if sample_tokens:
                    first_token = sample_tokens[0]
                    # Heuristic: if first token is 1 or >100000, it's likely BOS
                    if first_token == 1 or first_token >= 100000:
                        bos_token_id = first_token
            
            for idx, continuation in continuations:
                cont_enc = self._tokenizer.encode(continuation)
                cont_tokens = list(cont_enc.input_ids.data[0]) if len(cont_enc.input_ids.data) > 0 else []
                
                # Strip BOS token if present (tokenizer adds it when encoding separately)
                if cont_tokens and bos_token_id is not None and cont_tokens[0] == bos_token_id:
                    cont_tokens = cont_tokens[1:]
                
                if len(cont_tokens) == 1:
                    single_token_continuations.append((idx, continuation, cont_tokens[0]))
                else:
                    multi_token_continuations.append((idx, continuation))
            
            if single_token_continuations:
                try:
                    token_ids = [token_id for _, _, token_id in single_token_continuations]
                    log_probs = self._pipeline.get_next_token_log_probs(context, token_ids)
                    
                    for (idx, continuation, token_id), log_prob in zip(single_token_continuations, log_probs):
                        res[idx] = (float(log_prob), True)
                except Exception as e:
                    eval_logger.error(f"Error with get_next_token_log_probs: {e}, falling back to echo mode")
                    # Fallback to echo mode for these requests
                    for idx, continuation, _ in single_token_continuations:
                        try:
                            whole_enc = self._tokenizer.encode(context + continuation)
                            result = self._pipeline.generate(whole_enc, generation_config)
                            log_probs = result.log_probs[0] if result.log_probs else []
                            
                            if len(log_probs) == 0:
                                eval_logger.warning("Received empty log_probs, returning fake value")
                                res[idx] = (-1.0, True)
                                continue

                            # Extract continuation log_probs
                            context_tokens = self._tokenizer.encode(context)
                            continuation_tokens = self._tokenizer.encode(continuation)
                            if len(continuation_tokens) > 0 and continuation_tokens[0] == self._detect_bos_token_id():
                                continuation_tokens = continuation_tokens[1:]

                            continuation_start_idx = len(context_tokens)
                            continuation_log_probs = log_probs[continuation_start_idx:continuation_start_idx+len(continuation_tokens)]

                            answer = sum(continuation_log_probs)
                            res[idx] = (answer, True)
                        except Exception as e2:
                            eval_logger.error(f"Error in fallback: {e2}")
                            res[idx] = (-1.0, True)
            else:
                # Move single-token continuations to multi-token list for echo mode processing
                if single_token_continuations:
                    for idx, continuation, _ in single_token_continuations:
                        multi_token_continuations.append((idx, continuation))
            
            # Process multi-token continuations with echo mode
            for idx, continuation in multi_token_continuations:
                try:
                    # Tokenize full text
                    whole_enc = self._tokenizer.encode(context + continuation)
                    
                    # Generate with echo mode
                    result = self._pipeline.generate(whole_enc, generation_config)
                    log_probs = result.log_probs[0] if result.log_probs else []
                    
                    if len(log_probs) == 0:
                        eval_logger.warning("Received empty log_probs, returning fake value")
                        res[idx] = (-1.0, True)
                        continue
                    
                    # Extract continuation log_probs
                    # log_probs[i] = P(token[i] | tokens[0:i-1])
                    # For continuation starting at index context_len, we need log_probs starting from context_len
                    context_tokens = self._tokenizer.encode(context)
                    continuation_tokens = self._tokenizer.encode(continuation)
                    # Account for BOS token if present
                    if len(continuation_tokens) > 0 and continuation_tokens[0] == self._detect_bos_token_id():
                        continuation_tokens = continuation_tokens[1:]
                    
                    continuation_start_idx = len(context_tokens)
                    continuation_log_probs = log_probs[continuation_start_idx:continuation_start_idx+len(continuation_tokens)]
                    
                    # Sum log probabilities for the continuation
                    answer = sum(continuation_log_probs)
                    
                    # For greedy decoding (do_sample=False)
                    is_greedy = True
                    
                    res[idx] = (answer, is_greedy)
                    
                except Exception as e:
                    eval_logger.error(f"Error computing loglikelihood: {e}")
                    res[idx] = (-1.0, True)
            
            # Update progress bar after processing all continuations for this context
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
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
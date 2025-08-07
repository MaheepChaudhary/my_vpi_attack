import os
import io
import time
import json
import asyncio
from typing import Optional, Sequence, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (ADD THIS)
client = OpenAI(api_key="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")

@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict] = field(default_factory=dict)


def openai_complete(
    prompt_lst: List[str],
    decoding_args: OpenAIDecodingArguments,
    model_name: str,
    batch_size: int = 5
) -> Tuple[List[str], List[str], int, float]:
    """
    Updated OpenAI completion function for v1.0+ API
    """
    request_start = time.time()
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    prediction_lst = []
    finish_reason_lst = []
    
    progress_bar = tqdm(total=len(prompt_lst))
    original_max_tokens = decoding_args.max_tokens
    
    i = 0
    while i < len(prompt_lst):
        batch_prompts = prompt_lst[i:i + batch_size]
        retry_count = 0
        max_retries = 5
        wait_base = 10
        
        while retry_count < max_retries:
            try:
                # Process each prompt in the batch
                for prompt in batch_prompts:
                    if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
                        messages = [{"role": "user", "content": prompt}]
                        
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=decoding_args.max_tokens,
                            temperature=decoding_args.temperature,
                            top_p=decoding_args.top_p,
                            n=decoding_args.n,
                            stop=decoding_args.stop,
                            presence_penalty=decoding_args.presence_penalty,
                            frequency_penalty=decoding_args.frequency_penalty,
                        )
                        
                        prediction = response.choices[0].message.content
                        finish_reason = response.choices[0].finish_reason
                        
                        total_tokens += response.usage.total_tokens
                        total_prompt_tokens += response.usage.prompt_tokens
                        total_completion_tokens += response.usage.completion_tokens
                        
                    elif model_name == 'text-davinci-003':
                        response = client.completions.create(
                            model=model_name,
                            prompt=prompt,
                            max_tokens=decoding_args.max_tokens,
                            temperature=decoding_args.temperature,
                            top_p=decoding_args.top_p,
                            n=decoding_args.n,
                            stop=decoding_args.stop,
                            presence_penalty=decoding_args.presence_penalty,
                            frequency_penalty=decoding_args.frequency_penalty,
                        )
                        
                        prediction = response.choices[0].text
                        finish_reason = response.choices[0].finish_reason
                        
                        total_tokens += response.usage.total_tokens
                        total_prompt_tokens += response.usage.prompt_tokens
                        total_completion_tokens += response.usage.completion_tokens
                    
                    prediction_lst.append(prediction)
                    finish_reason_lst.append(finish_reason)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.2)
                
                progress_bar.update(len(batch_prompts))
                i += batch_size
                
                # Reset retry parameters on success
                retry_count = 0
                wait_base = 10
                decoding_args.max_tokens = original_max_tokens
                break
                
            except Exception as e:  # FIXED ERROR HANDLING
                print(f"Error: {repr(e)}")
                retry_count += 1
                print(f"Batch error: {i} to {i + batch_size}")
                print(f"Retry number: {retry_count}")
                
                if "Please reduce" in str(e) or "maximum context length" in str(e):
                    decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                    print(f"Reducing target length to {decoding_args.max_tokens}, Retrying...")
                elif "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    print(f"Hit request rate limit; retrying...; sleep ({wait_base})")
                    time.sleep(wait_base)
                    wait_base = min(wait_base * 2, 300)
                else:
                    print(f"Unknown error, waiting {wait_base} seconds...")
                    time.sleep(wait_base)
                    wait_base = min(wait_base * 2, 300)
                
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} retries, skipping batch")
                    for _ in range(len(batch_prompts)):
                        prediction_lst.append("")
                        finish_reason_lst.append("error")
                    progress_bar.update(len(batch_prompts))
                    i += batch_size
                    break
    
    progress_bar.close()
    request_duration = time.time() - request_start
    print(f"Generated {len(prediction_lst)} responses in {request_duration:.2f}s")
    
    # Calculate cost
    if model_name.startswith("gpt-3.5-turbo"):
        cost = (0.0015 * total_prompt_tokens + 0.002 * total_completion_tokens) / 1000
    elif model_name.startswith("gpt-4"):
        cost = (0.03 * total_prompt_tokens + 0.06 * total_completion_tokens) / 1000
    elif model_name == 'text-davinci-003':
        cost = 0.02 * total_tokens / 1000
    else:
        cost = 0
    
    return prediction_lst, finish_reason_lst, total_tokens, cost


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
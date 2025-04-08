import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_1b-250129/400/', trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained('/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_1b-250129/400/', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-1_8b-chat/snapshots/763507996f322ce22c3e08ed22737fb63ef6613c/', trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained('/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-1_8b-chat/snapshots/763507996f322ce22c3e08ed22737fb63ef6613c/', trust_remote_code=True)
model = model.eval()

input_ids = torch.tensor(tokenizer(["hello"], padding=False, truncation=False)['input_ids'])

outputs = model.generate(input_ids=input_ids, max_new_tokens=50)

decodeds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decodeds)


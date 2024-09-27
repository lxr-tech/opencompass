from opencompass.models import TurboMindModelLong

path_dict = {

             'sdm_24090101': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500/52500_hf', 
             'sdm_24090102': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500_NeedleFix_hf/', 
             'sdm_24090103': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_3.0.0_256K_hf/original/', 
             'sdm_24090101d2': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500/dynamic', 
             'sdm_24090102d2': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500_NeedleFix_hf/dynamic', 
             'sdm_24090103d2': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_3.0.0_256K_hf/dynamic/', 
             'sdm_24090102d3': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500_NeedleFix_hf/dynamic_v2', 
             'sdm_24090103d3': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/official_Ampere2.5_7B_3.0.0_256K_hf/dynamic_v2/', 

             'sdm_24090601a': '/cpfs01/shared/public/liuxiaoran/ckpt_from_demin/volc_official_Ampere2.5_7B_Enhance_256k.0.0.0_FixBBHLeak_wsd_from_50000_to_52500_NeedleFix_fsp_s2_internlm2_5_420_hf/', 
             "sdm_24091001": "/cpfs02/puyu/shared/alillm2/alillm2/songdemin/ckpts/official_Ampere2.5_7B_LongContext_2.0.1_FT_s2_1m_hf/420"
            }

num_gpus = {'llama2_7B': 4, 'llama2_7B_chat': 4, 'llama2_13B': 8, 
            'llama3_8B': 4, 'llama3_8B_chat': 4, 'llama3_1_8B': 4, 'llama3_1_8B_chat': 4, 'llama3_1_70B': 8, 
            'qwen1.5_1B': 4, 'qwen1.5_7B': 4, 'qwen1.5_14B': 8, 'qwen1.5_32B': 8, 'internlm2.5_7B_enhance': 4, 
            'qwen2_1B': 4, 'qwen2_1B_chat': 4, 'qwen2_7B': 4, 'qwen2_7B_chat': 4, 'qwen2_72B': 8, 
            'internlm2_7B': 4, 'internlm2_7B_chat': 4, 
            'internlm2.5_7B': 4, 'internlm2.5_7B_chat': 4, 
            'internlm2_1B': 4, 'internlm2_20B': 4, 
            'mistral3_7B': 4, 'mistral3_7B_chat': 4, 
            'glm4_9B_chat_1M': 4, 
            }

tags = [
        ('', 'sdm_24090102d3', -1, 
         dict(session_len=1124000, max_batch_size=1, cache_max_entry_count=0.5, tp=8)), 
        ('', 'sdm_24090103d3', -1, 
         dict(session_len=1124000, max_batch_size=1, cache_max_entry_count=0.5, tp=8)), 

        ('', 'sdm_24090601a', -1, 
         dict(session_len=1124000, max_batch_size=1, cache_max_entry_count=0.5, tp=8)), 

    ]

models = []

for abbr, group, cat_len, engine_config in tags:
    models.append(
        dict(
            type=TurboMindModelLong,
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            engine_config=engine_config,
            gen_config=dict(top_k=1, top_p=1, temperature=1.0, max_new_tokens=500),
            max_out_len=50,
            max_seq_len=1048576,
            batch_size=1,
            concurrency=1,
            run_cfg=dict(num_gpus=1,  # num_gpus[group.split('-')[0]], 
                         num_procs=1),
            end_str='<eoa>',
        )
    )
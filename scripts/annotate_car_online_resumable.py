#!/usr/bin/env python3
"""Car-only retry adapter preserving strict labels and the frozen annotator."""
from __future__ import annotations

import json
import time

import annotate_multiclass_online_sensenova as online

original_post_json = online.post_json
original_request_one = online.request_one


def resilient_post_json(url, body, headers, timeout):
    for attempt in range(1, 4):
        try:
            return original_post_json(url, body, headers, timeout)
        except (OSError, TimeoutError) as error:
            print(f'[car transport retry] {attempt}/3 {type(error).__name__}', flush=True)
            if attempt == 3:
                raise
            time.sleep(5 * attempt)


def request_one(config, prompt, row, schema, raw_dir, limiter):
    arrays = [item['feature_id'] for item in schema['universal_features'] + schema.get('category_specific_features', [])
              if item['value_type'] == 'set']
    prompt += ('\n输出格式约束（不改变状态定义）：整条回复只能包含一个完整JSON对象，结束大括号后不能再输出第二个对象。'
               '\n以下set字段必须是JSON数组，不能是字符串：' + json.dumps(arrays, ensure_ascii=False)
               + '\n例如occluder_type无可见遮挡必须为["none"]，无法判断必须为["unknown"]；不能写成"none"或"unknown"。'
               '\nvisible_vehicle_parts也必须为数组，例如["front","wheels"]或["unknown"]。'
               '\n继续严格使用同一schema全部枚举和视觉证据，不增加或删除任何字段。')
    return original_request_one(config, prompt, row, schema, raw_dir, limiter)


if __name__ == '__main__':
    online.post_json = resilient_post_json
    online.request_one = request_one
    online.main()

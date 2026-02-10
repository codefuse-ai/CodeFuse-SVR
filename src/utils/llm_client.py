import json

import requests


def send_chat_completion(
        api_key,
        base_url: str,
        model_name: str,
        user_prompt: str,
        system_prompt,
        image_url: str = None,
        temperature: float = 0.,
        stream=False,
        trace_id=None
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "SOFA-TraceId": trace_id if trace_id else "1111",
        "SOFA-RpcId": "0.1"
    }

    if not image_url:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt, },
                {"role": "user", "content": user_prompt}
            ],
            "stream": stream,
            "temperature": temperature,
            # "top_k": -1,
            # "top_p": 0.95,
            # "chat_template_kwargs": {"enable_thinking": False}
        }
    else:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ],
            "stream": stream,
            "temperature": temperature,
            # "chat_template_kwargs": {"enable_thinking": False}
        }

    response = requests.post(base_url, headers=headers, json=payload, proxies={})
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    # Main execution code
    import yaml
    from prompt.svr_prompt import SVR_PROMPT
    from src.utils.image_utils import image_to_base64, image_url_to_base64


    with open("config/config.yaml", "r") as infile:
        conf = yaml.safe_load(infile)

    model_name = "Qwen3-VL-235B-A22B-Instruct"
    api_key = "your-api-key "

    img_file = "/path/to/image.png"
    img_str = image_to_base64(img_file)

    with open("image_test.txt", "w") as outf:
        outf.write(img_str)

    print("image string: ", img_str[:30])
    msg = [
        {"role": "system", "content": SVR_PROMPT["system_prompt"]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_str}},
            {"type": "text", "text": SVR_PROMPT["user_prompt"]}
        ]}
    ]


    response  = "<category>\nWebpage Screenshot\n</category>\n<content>\nThe image converted to HTML code is as follows:\n```html\n<!DOCTYPE html>\n<html>\n<head>\n <title>Store Settings</title>\n</head>\n<body>\n <header>\n <nav>\n <div class=\"logo\">My Sites</div>\n <div class=\"reader\">Reader</div>\n <div class=\"write\">Write</div>\n <div class=\"user-profile\">User Profile</div>\n </nav>\n </header>\n <aside>\n <ul>\n <li>Dashboard</li>\n <li>Products</li>\n <li>Orders</li>\n <li>Promotions</li>\n <li>Reviews</li>\n <li class=\"active\">Settings</li>\n </ul>\n </aside>\n <main>\n <div class=\"site-info\">\n <div class=\"site-name\">Allendav's Store Test Site</div>\n <div class=\"settings\">Settings / Email</div>\n </div>\n <div class=\"content\">\n <div class=\"tabs\">\n <div class=\"tab\">Payments</div>\n <div class=\"tab\">Shipping</div>\n <div class=\"tab\">Taxes</div>\n <div class=\"tab active\">Email</div>\n </div>\n <div class=\"email-settings\">\n <div class=\"origin\">Origin</div>\n <div class=\"from-name\">\n <label>From name</label>\n <input type=\"text\" value=\"Allendav&#039;s Store Test Site\">\n <p>Emails will appear in recipients inboxes 'from' this name.</p>\n </div>\n <div class=\"from-address\">\n <label>From address</label>\n <input type=\"text\" value=\"************@gmail.com\">\n <p>If recipients reply to store emails they will be sent to this address.</p>\n </div>\n <div class=\"internal-notifications\">\n <div class=\"title\">Internal notifications</div>\n <p>Email notifications sent to store staff.</p>\n </div>\n </div>\n </div>\n </main>\n</body>\n</html>\n```\n</content>"
    print(response)
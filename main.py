import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from generators import GeneratorBase, StarCoder
import json

from util import logger, get_parser

app = FastAPI()
app.add_middleware(CORSMiddleware)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    generated_text: str = generator.generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200,
    }


# Add compatibility for CodeGeeX vscode extension
@app.post("/api/v2/multilingual_code_generate")
@app.post("/api/v2/multilingual_code_generate_adapt")
@app.post("/api/v2/multilingual_code_generate_block")
async def api(request: Request):
    json_request: dict = await request.json()
    prefix: str = json_request.pop("prompt")
    suffix: str = json_request.pop("suffix")
    prompt: str = prefix if suffix == "" else f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    json_request["max_new_tokens"] = 32
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(prompt)}')
    generated_text: str = generator.generate(prompt, json_request)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "status": 0,
        "message": "success",
        "result": {
            "app": "multilingual_code_generate",
            "input": json_request,
            "output": {
                "errcode": 0,
                "code": [generated_text[len(prompt):]],
            },
        },
    }


def main():
    global generator
    args = get_parser().parse_args()
    generator = StarCoder(args.pretrained, device_map='auto')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

import json
import base64
import io
import yaml
from model_loader import ModelLoader


def init_context(context):
    context.logger.info("Init context...  0%")
    model_path = "/opt/nuclio/model"
    model_handler = ModelLoader(model_path)
    context.user_data.model_handler = model_handler

    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        function_config = yaml.safe_load(function_file)

    labels_spec = function_config['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    context.user_data.labels = labels

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run defect_detection on Nugg's model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))

    intermediate_results = context.user_data.model_handler.infer(buf)
    final_result = context.user_data.model_handler.extraction(intermediate_results, context.user_data.labels)

    return context.Response(body=json.dumps(final_result), headers={},
                            content_type='application/json', status_code=200)

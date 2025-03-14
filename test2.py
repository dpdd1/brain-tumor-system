from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import requests
import time
import logging

# 初始化Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
classification_model = YOLO('D:/demo01/Myproject/yolo11s_model.pt')
malignancy_model = YOLO('D:/demo01/Myproject/models/detection.pt')

# 类别映射
CLASS_MAPPING = {
    'Glioma': '胶质瘤',
    'Meningioma': '脑膜瘤',
    'No Tumor': '无脑肿瘤',
    'Pituitary': '垂体瘤'
}

MALIGNANCY_MAPPING = {
    'positive': '良性',
    'negative': '恶性',
    'Positive': '良性',
    'Negative': '恶性'
}

# ==================== 大模型配置 ====================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

API_KEYS = {
    "deepseek": "sk-d1e6cbd1eb4b4551b777a190ab1db1f8",
    "qwen": "sk-30a1044e830848bc8dd196d3b60b178d"
}

HEADERS = {
    "deepseek": {
        "Authorization": f"Bearer {API_KEYS['deepseek']}",
        "Content-Type": "application/json"
    },
    "qwen": {
        "Authorization": f"Bearer {API_KEYS['qwen']}",
        "Content-Type": "application/json",
        "X-DashScope-Service-Id": "1942264"
    }
}

MODEL_CONFIG = {
    "deepseek": {
        "model_name": "deepseek-chat",
        "temperature": 0.3
    },
    "qwen": {
        "model_name": "qwen-max",
        "temperature": 0.5
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ai_response(model_type, class_info, detection_info):
    """通用大模型响应获取"""
    max_retries = 3
    retry_delay = 1

    prompt = f"""你是一名专业的放射科医生，根据以下脑肿瘤检测结果进行分析：
1. 肿瘤分类结果：类型为【{class_info['class_name']}】，置信度{class_info['probability']:.0f}%
2. 良恶性检测：良性概率{detection_info['malignant_prob']:.0f}%，恶性概率{detection_info['benign_prob']:.0f}%
3. 肿瘤位置信息：{detection_info['boxes']}

请结合医学知识分析预测：
- 该类型肿瘤的典型发展规律
- 基于当前检测结果的潜在风险
- 建议的随访周期和检查方式
- 日常注意事项

用中文回答，分点说明，控制在300字以内。"""

    payload = {
        "model": MODEL_CONFIG[model_type]["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": MODEL_CONFIG[model_type]["temperature"]
    }

    for attempt in range(max_retries):
        try:
            if model_type == "qwen":
                response = requests.post(
                    QWEN_API_URL,
                    headers=HEADERS["qwen"],
                    json={
                        "model": payload["model"],
                        "input": {"messages": payload["messages"]},
                        "parameters": {"temperature": payload["temperature"]}
                    },
                    timeout=30
                )
            else:
                response = requests.post(
                    DEEPSEEK_API_URL,
                    headers=HEADERS["deepseek"],
                    json=payload,
                    timeout=30
                )

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                time.sleep(retry_after)
                continue

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip() if model_type != "qwen" \
                    else response.json()['output']['text'].strip()

            logger.warning(f"{model_type} API异常: {response.status_code} {response.text}")

        except Exception as e:
            logger.error(f"{model_type} API尝试 {attempt + 1} 失败: {str(e)}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return f"{model_type}医学分析暂时不可用，请稍后再试"

def get_threshold(param_name, default=50):
    """获取并验证阈值参数"""
    try:
        value = float(request.form.get(param_name, default))
        return max(0, min(100, value)) / 100  # 转换为0-1范围
    except:
        return default / 100

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return {'error': '未上传文件'}, 400

    file = request.files['file']
    if not file or file.filename == '':
        return {'error': '未选择文件'}, 400

    if not allowed_file(file.filename):
        return {'error': '不支持的文件类型'}, 400

    # 获取阈值参数
    cls_threshold = get_threshold('cls_threshold')
    det_threshold = get_threshold('det_threshold')

    upload_path = None
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # 执行预测
        cls_results = classification_model(upload_path)
        mal_results = malignancy_model(upload_path)

        output = {
            "detections": [],
            "classifications": [],
            "malignancy": [],
            "result_image": "",
            "medical_analysis": "",
            "thresholds": {
                "classification": int(cls_threshold * 100),  # 转换为整数
                "detection": int(det_threshold * 100)       # 转换为整数
            }
        }

        # 处理分类模型结果
        def process_classification(result):
            if result.boxes:
                for box in result.boxes:
                    if box.conf >= cls_threshold:
                        class_en = classification_model.names[int(box.cls)]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        output["detections"].append({
                            "class": CLASS_MAPPING.get(class_en, class_en),
                            "confidence": float(box.conf.item()),
                            "display_conf": f"{box.conf.item():.0%}",  # 整数百分比
                            "position": {
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2)
                            }
                        })
            if result.probs:
                for class_id, prob in enumerate(result.probs.data.tolist()):
                    if prob >= cls_threshold:
                        class_en = classification_model.names[class_id]
                        output["classifications"].append({
                            "class": CLASS_MAPPING.get(class_en, class_en),
                            "probability": float(prob),
                            "display_prob": f"{prob:.0%}"  # 整数百分比
                        })

        # 处理良恶性检测结果
        def process_malignancy(result):
            if result.boxes:
                for box in result.boxes:
                    if box.conf >= det_threshold:
                        class_en = malignancy_model.names[int(box.cls)]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        output["malignancy"].append({
                            "class": MALIGNANCY_MAPPING.get(class_en, class_en),
                            "probability": float(box.conf.item()),
                            "display_prob": f"{box.conf.item():.0%}",  # 整数百分比
                            "position": {
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2)
                            }
                        })
            if result.probs:
                for class_id, prob in enumerate(result.probs.data.tolist()):
                    if prob >= det_threshold:
                        class_en = malignancy_model.names[class_id]
                        output["malignancy"].append({
                            "class": MALIGNANCY_MAPPING.get(class_en, class_en),
                            "probability": float(prob),
                            "display_prob": f"{prob:.0%}"  # 整数百分比
                        })

        # 处理所有结果
        for result in cls_results:
            process_classification(result)
        for result in mal_results:
            process_malignancy(result)

        # 生成结果图像
        if cls_results and output["detections"]:
            result_img = cls_results[0].plot()
            result_filename = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            Image.fromarray(result_img).save(result_path)
            output["result_image"] = result_filename

        # 获取AI分析
        model_type = request.form.get('model', 'deepseek').lower()
        if model_type not in ['deepseek', 'qwen', 'basic']:
            model_type = 'deepseek'

        if model_type != 'basic' and output['detections']:
            try:
                malignant_probs = [m['probability'] for m in output['malignancy'] if m['class'] == '良性']
                benign_probs = [m['probability'] for m in output['malignancy'] if m['class'] == '恶性']

                analysis_data = {
                    "class_info": {
                        "class_name": output['detections'][0]['class'],
                        "probability": output['detections'][0]['confidence'] * 100
                    },
                    "detection_info": {
                        "malignant_prob": max(malignant_probs)*100 if malignant_probs else 0,
                        "benign_prob": max(benign_probs)*100 if benign_probs else 0,
                        "boxes": [
                            f"({d['position']['x1']}, {d['position']['y1']})→({d['position']['x2']}, {d['position']['y2']})"
                            for d in output['detections']]
                    }
                }

                output['medical_analysis'] = get_ai_response(
                    model_type=model_type,
                    class_info=analysis_data['class_info'],
                    detection_info=analysis_data['detection_info']
                )
            except Exception as e:
                logger.error(f"AI分析失败: {str(e)}")
                output['medical_analysis'] = "医学分析生成失败"

        return output

    except Exception as e:
        logger.error(f"系统错误: {str(e)}")
        return {'error': f"分析失败: {str(e)}"}, 500
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)
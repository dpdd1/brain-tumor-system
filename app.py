from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sqlite3
import hashlib
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from ultralytics import YOLO
import uuid
import shutil
import requests
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于session加密

# 配置上传文件存储路径
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# 确保上传和结果目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 加载YOLOv11模型
detection_model = None
segmentation_model = None
classification_model = None

# 星火大模型配置
SPARK_APP_ID = 'b1968706'
SPARK_API_KEY = 'e6a007a32da370f149c75b34d10cdd74'
SPARK_API_SECRET = 'NTFmZGFlMThkMTQyNWZjZDcyZDAxOThm'

# DeepSeek配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-d1e6cbd1eb4b4551b777a190ab1db1f8"
DEEPSEEK_HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

def load_models():
    global detection_model, segmentation_model, classification_model
    try:
        detection_model = YOLO('models/detection.pt')
        segmentation_model = YOLO('models/segmentation.pt')
        classification_model = YOLO('models/classification.pt')
        print("YOLOv11模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")

# 在应用启动时加载模型
load_models()

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
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

API_KEYS = {
    "deepseek": DEEPSEEK_API_KEY,
    "qwen": "sk-30a1044e830848bc8dd196d3b60b178d"
}

HEADERS = {
    "deepseek": DEEPSEEK_HEADERS,
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

# 初始化数据库
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # 创建治疗记录表
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            initial_image TEXT NOT NULL,
            mask_image TEXT NOT NULL,
            detection_date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

# 确保数据库表存在
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 登录页面路由
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 对密码进行哈希处理
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # 连接数据库验证用户
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('diagnosis'))
        else:
            flash('用户名或密码错误')
    return render_template('login.html')

# 注册路由
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    
    # 对密码进行哈希处理
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        flash('注册成功，请登录')
    except sqlite3.IntegrityError:
        flash('用户名已存在')
    except Exception as e:
        flash('注册失败，请稍后重试')
    
    return redirect(url_for('login'))

# 登出路由
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# 肿瘤诊断页面
@app.route('/diagnosis')
def diagnosis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('diagnosis.html')

# 处理诊断图像上传和检测
@app.route('/upload_diagnosis', methods=['POST'])
def upload_diagnosis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('diagnosis'))
    
    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('diagnosis'))
    
    if file and allowed_file(file.filename):
        try:
            # 保存上传的图像
            filename = secure_filename(f"{session.get('username')}_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 使用YOLOv11检测模型进行推理
            results = detection_model(file_path)
            
            # 处理检测结果
            result_img = results[0].plot()
            result_filename = f"detection_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img)
            
            # 获取检测结果信息
            boxes = results[0].boxes
            has_tumor = len(boxes) > 0
            
            # 准备返回数据
            result_data = {
                'has_tumor': has_tumor,
                'confidence': float(boxes.conf[0]) * 100 if has_tumor else 0,
                'original_image': f"/static/uploads/{filename}",
                'result_image': f"/static/results/{result_filename}"
            }
            
            # 根据检测结果生成建议
            if has_tumor:
                if result_data['confidence'] > 80:
                    result_data['recommendation'] = "检测到高置信度肿瘤，建议立即就医进行进一步检查和治疗。"
                else:
                    result_data['recommendation'] = "检测到可能的肿瘤，建议进行进一步的医学检查确认。"
            else:
                result_data['recommendation'] = "未检测到肿瘤，建议定期复查。"
            
            return render_template('diagnosis.html', result=result_data)
            
        except Exception as e:
            flash(f'处理图像时出错: {str(e)}')
            return redirect(url_for('diagnosis'))
    else:
        flash('不支持的文件格式')
        return redirect(url_for('diagnosis'))

# 肿瘤分割页面
@app.route('/segmentation')
def segmentation():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('segmentation.html')

# 处理分割图像上传和分割
@app.route('/upload_segmentation', methods=['POST'])
def upload_segmentation():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('segmentation'))
    
    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('segmentation'))
    
    if file and allowed_file(file.filename):
        try:
            # 保存上传的图像
            filename = secure_filename(f"{session.get('username')}_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 使用YOLOv11分割模型进行推理
            results = segmentation_model(file_path)
            
            # 处理分割结果
            # 1. 保存分割结果图像
            result_img = results[0].plot()
            segmentation_filename = f"segmentation_{uuid.uuid4().hex}.jpg"
            segmentation_path = os.path.join(app.config['RESULTS_FOLDER'], segmentation_filename)
            cv2.imwrite(segmentation_path, result_img)
            
            # 2. 生成掩码图像
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                # 获取原始图像
                original_img = cv2.imread(file_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # 获取掩码
                masks = results[0].masks.data.cpu().numpy()
                mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
                
                for i in range(len(masks)):
                    mask_i = masks[i].astype(np.uint8) * 255
                    mask_i = cv2.resize(mask_i, (original_img.shape[1], original_img.shape[0]))
                    mask = cv2.bitwise_or(mask, mask_i)
                
                # 保存掩码图像
                mask_filename = f"mask_{uuid.uuid4().hex}.jpg"
                mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
                cv2.imwrite(mask_path, mask)
                
                # 3. 生成叠加图像
                # 创建彩色掩码用于叠加
                colored_mask = np.zeros_like(original_img)
                colored_mask[:,:,0] = mask  # 红色通道
                
                # 叠加原始图像和掩码
                alpha = 0.5
                overlay_img = cv2.addWeighted(original_img, 1, colored_mask, alpha, 0)
                
                # 保存叠加图像
                overlay_filename = f"overlay_{uuid.uuid4().hex}.jpg"
                overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                
                # 自动保存到治疗记录
                try:
                    username = session.get('username')
                    detection_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # 保存记录到数据库
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute('INSERT INTO records (username, initial_image, mask_image, detection_date, created_at) VALUES (?, ?, ?, ?, ?)',
                             (username, f"uploads/{filename}", f"results/{mask_filename}", detection_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    conn.close()
                    
                    flash('分割结果已自动保存到治疗记录')
                except Exception as e:
                    print(f"保存记录失败: {str(e)}")
                
                # 准备返回数据
                result_data = {
                    'original_image': f"/static/uploads/{filename}",
                    'segmentation_image': f"/static/results/{segmentation_filename}",
                    'mask_image': f"/static/results/{mask_filename}",
                    'overlay_image': f"/static/results/{overlay_filename}"
                }
            else:
                # 如果没有检测到掩码，返回默认图像
                result_data = {
                    'original_image': f"/static/uploads/{filename}",
                    'segmentation_image': f"/static/results/{segmentation_filename}",
                    'mask_image': "/static/images/mask-placeholder.png",
                    'overlay_image': "/static/images/overlay-placeholder.png",
                    'error': "未检测到肿瘤区域"
                }
            
            return render_template('segmentation.html', result=result_data)
            
        except Exception as e:
            flash(f'处理图像时出错: {str(e)}')
            return redirect(url_for('segmentation'))
    else:
        flash('不支持的文件格式')
        return redirect(url_for('segmentation'))

# 图像预处理函数：高斯滤波
def apply_gaussian_blur(image, kernel_size):
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    # 增加sigma值以产生更明显的模糊效果
    sigma = kernel_size / 3.0
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# 图像预处理函数：对比度调整
def adjust_contrast(image, alpha):
    # alpha > 1 增加对比度，alpha < 1 降低对比度
    # 添加beta值以调整亮度，使对比度变化更明显
    beta = 128 * (1 - alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 调整分割阈值和图像预处理
@app.route('/adjust_image', methods=['POST'])
def adjust_image():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 50))
        gaussian_blur = data.get('gaussian_blur', 0)  # 高斯滤波核大小
        contrast = data.get('contrast', 100)  # 对比度调整值
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(os.path.join('static', image_path.lstrip('/'))):
            return jsonify({'success': False, 'message': '图像不存在'})
        
        # 重新处理图像，使用更敏感的阈值范围
        threshold_value = max(0.05, min(0.95, threshold / 100.0))  # 将阈值映射到0.05-0.95的范围
        file_path = os.path.join('static', image_path.lstrip('/'))
        
        # 读取原始图像
        original_img = cv2.imread(file_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 应用图像预处理
        processed_img = original_img.copy()
        
        # 应用高斯滤波
        if gaussian_blur > 0:
            kernel_size = 2 * gaussian_blur + 1  # 确保是奇数
            processed_img = apply_gaussian_blur(processed_img, kernel_size)
        
        # 应用对比度调整
        contrast_alpha = contrast / 100.0  # 将百分比转换为alpha值
        if contrast != 100:  # 如果不是默认值100%
            processed_img = adjust_contrast(processed_img, contrast_alpha)
        
        # 保存预处理后的图像
        preprocessed_filename = f"preprocessed_{uuid.uuid4().hex}.jpg"
        preprocessed_path = os.path.join(app.config['RESULTS_FOLDER'], preprocessed_filename)
        cv2.imwrite(preprocessed_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        
        # 使用YOLOv11分割模型进行推理，设置置信度阈值
        results = segmentation_model(preprocessed_path, conf=threshold_value)
        
        # 处理分割结果
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            # 获取掩码
            masks = results[0].masks.data.cpu().numpy()
            mask = np.zeros((processed_img.shape[0], processed_img.shape[1]), dtype=np.uint8)
            
            for i in range(len(masks)):
                mask_i = masks[i].astype(np.uint8) * 255
                mask_i = cv2.resize(mask_i, (processed_img.shape[1], processed_img.shape[0]))
                mask = cv2.bitwise_or(mask, mask_i)
                
            # 创建彩色掩码用于可视化
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # 创建叠加图像
            overlay = cv2.addWeighted(cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR), 0.7, colored_mask, 0.3, 0)
            
            # 保存分割结果
            mask_filename = f"segmentation_{uuid.uuid4().hex}.jpg"
            overlay_filename = f"overlay_{uuid.uuid4().hex}.jpg"
            
            mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
            overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
            
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(overlay_path, overlay)
            
            # 返回结果
            return jsonify({
                'success': True,
                'mask_image': mask_filename,
                'overlay_image': overlay_filename,
                'preprocessed_image': preprocessed_filename
            })
        else:
            return jsonify({'success': False, 'message': '未检测到肿瘤区域'})
    except Exception as e:
        print(f"分割处理错误: {str(e)}")
        return jsonify({'success': False, 'message': f'处理失败: {str(e)}'})

# 新增的AI分析功能
def get_ai_response(model_type, class_info, detection_info):
    """通用大模型响应获取"""
    max_retries = 3
    retry_delay = 1

    # 构造提示词
    prompt = f"""你是一名专业的放射科医生，根据以下脑肿瘤检测结果进行分析：
1. 肿瘤分类结果：类型为【{class_info['class_name']}】，置信度{class_info['probability']:.1f}%
2. 良恶性检测：良性概率{detection_info['malignant_prob']:.1f}%，恶性概率{detection_info['benign_prob']:.1f}%
3. 肿瘤位置信息：{detection_info['boxes']}

请结合医学知识分析预测：
- 该类型肿瘤的典型发展规律
- 基于当前检测结果的潜在风险
- 建议的随访周期和检查方式
- 日常注意事项

用中文回答，分点说明，控制在300字以内。"""

    # 构造请求参数
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
                if model_type == "qwen":
                    return response.json()['output']['text'].strip()
                else:
                    return response.json()['choices'][0]['message']['content'].strip()

            print(f"{model_type} API异常: {response.status_code} {response.text}")

        except Exception as e:
            print(f"{model_type} API尝试 {attempt + 1} 失败: {str(e)}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return f"{model_type}医学分析暂时不可用，请稍后再试"

# 新增的脑部检测API
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return {'error': '未上传文件'}, 400

    file = request.files['file']
    if not file or file.filename == '':
        return {'error': '未选择文件'}, 400

    if not allowed_file(file.filename):
        return {'error': '不支持的文件类型'}, 400

    upload_path = None
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # 执行预测
        cls_results = classification_model(upload_path)
        mal_results = detection_model(upload_path)

        output = {
            "detections": [],
            "classifications": [],
            "malignancy": [],
            "result_image": "",
            "medical_analysis": ""
        }

        # 分类模型结果处理
        for result in cls_results:
            if result.boxes:
                for box in result.boxes:
                    class_en = classification_model.names[int(box.cls)]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    output["detections"].append({
                        "class": CLASS_MAPPING.get(class_en, class_en),
                        "confidence": float(box.conf.item()),
                        "display_conf": f"{box.conf.item():.2%}",
                        "position": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                    })
            elif result.probs:
                for class_id, prob in result.probs.top4:
                    class_en = classification_model.names[class_id]
                    output["classifications"].append({
                        "class": CLASS_MAPPING.get(class_en, class_en),
                        "probability": float(prob),
                        "display_prob": f"{prob:.2%}"
                    })

        # 良恶性检测处理
        for result in mal_results:
            if result.probs:
                # 直接获取所有分类概率
                probs = result.probs.data.tolist()
                for class_id, prob in enumerate(probs):
                    class_en = detection_model.names[class_id]
                    output["malignancy"].append({
                        "class": MALIGNANCY_MAPPING.get(class_en, class_en),
                        "probability": float(prob),
                        "display_prob": f"{prob:.2%}"
                    })
            elif result.boxes:
                # 处理检测框结果
                for box in result.boxes:
                    class_en = detection_model.names[int(box.cls)]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    output["malignancy"].append({
                        "class": MALIGNANCY_MAPPING.get(class_en, class_en),
                        "probability": float(box.conf.item()),
                        "display_prob": f"{box.conf.item():.2%}",
                        "position": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                    })

        # 生成结果图像
        if cls_results:
            result_img = cls_results[0].plot()
            result_filename = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            Image.fromarray(result_img).save(result_path)
            output["result_image"] = result_filename

        # 获取AI分析结果
        model_type = request.form.get('model', 'deepseek').lower()
        if model_type not in ['deepseek', 'qwen']:
            model_type = 'deepseek'

        if output['detections']:
            # 计算良恶性概率
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

            try:
                output['medical_analysis'] = get_ai_response(
                    model_type=model_type,
                    class_info=analysis_data['class_info'],
                    detection_info=analysis_data['detection_info']
                )
            except Exception as e:
                print(f"AI分析失败: {str(e)}")
                output['medical_analysis'] = "医学分析生成失败"

        return output
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 添加访问上传文件的路由
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 保留原来的调整阈值接口以保持兼容性
@app.route('/adjust_threshold', methods=['POST'])
def adjust_threshold():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        data = request.get_json()
        threshold = data.get('threshold', 50)
        image_path = data.get('image_path')
        
        # 调用图像处理函数，传递完整的参数
        response = adjust_image()
        return response
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 脑部检测页面
@app.route('/analysis')
def analysis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('brain_analysis'))

# 新的脑部检测页面 - 重定向到新的分析系统
@app.route('/brain_analysis')
def brain_analysis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index1.html')



# 导出分析PDF
@app.route('/export_analysis_pdf')
def export_analysis_pdf():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from io import BytesIO
        
        # 创建PDF内存流
        buffer = BytesIO()
        
        # 创建PDF文档
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # 添加标题
        title_style = styles['Heading1']
        title = Paragraph("脑肿瘤医学诊断平台 - 分析报告", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.25*inch))
        
        # 添加患者信息
        patient_info = [
            ["患者ID:", session.get('username')],
            ["分析日期:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        t = Table(patient_info, colWidths=[1.5*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.25*inch))
        
        # 添加分析结果
        if 'result' in session:
            result = session.get('result', {})
            
            # 添加结果标题
            result_title = Paragraph("分析结果", styles['Heading2'])
            elements.append(result_title)
            elements.append(Spacer(1, 0.1*inch))
            
            # 添加结果表格
            result_data = [
                ["识别结果", "置信度", "处理时间"],
                [result.get('recognition', 'N/A'), 
                 result.get('probability', 'N/A'), 
                 result.get('time', 'N/A')]
            ]
            
            t = Table(result_data, colWidths=[2*inch, 2*inch, 2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.25*inch))
            
            # 添加AI分析结果
            if 'ai_analysis' in result and result['ai_analysis']:
                ai_title = Paragraph("AI分析", styles['Heading2'])
                elements.append(ai_title)
                elements.append(Spacer(1, 0.1*inch))
                
                ai_text = Paragraph(result['ai_analysis'], styles['Normal'])
                elements.append(ai_text)
                elements.append(Spacer(1, 0.25*inch))
            
            # 添加图像
            if 'result_image' in result:
                img_path = os.path.join(os.getcwd(), 'static', result['result_image'].lstrip('/'))
                if os.path.exists(img_path):
                    img = Image(img_path, width=6*inch, height=4*inch)
                    elements.append(img)
        
        # 构建PDF
        doc.build(elements)
        
        # 获取PDF内容
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # 创建响应
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=analysis_report.pdf'
        
        return response
    except Exception as e:
        flash(f'生成PDF时出错: {str(e)}')
        return redirect(url_for('analysis'))

# 治疗记录页面
@app.route('/records')
def records():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # 获取当前用户的治疗记录
    username = session.get('username')
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  # 启用行工厂，使结果可以通过列名访问
    c = conn.cursor()
    c.execute('SELECT * FROM records WHERE username = ? ORDER BY detection_date DESC', (username,))
    records = c.fetchall()
    conn.close()
    
    return render_template('records.html', records=records)

# 添加治疗记录
@app.route('/add_record', methods=['POST'])
def add_record():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        username = session.get('username')
        detection_date = request.form['detection_date']
        
        # 处理初始图片上传
        initial_image = request.files['initial_image']
        if initial_image and allowed_file(initial_image.filename):
            initial_filename = secure_filename(f"{username}_{int(datetime.now().timestamp())}_initial.{initial_image.filename.rsplit('.', 1)[1].lower()}")
            initial_image_path = os.path.join(app.config['UPLOAD_FOLDER'], initial_filename)
            initial_image.save(initial_image_path)
            initial_image_rel_path = f"uploads/{initial_filename}"
        else:
            flash('初始图片格式不正确')
            return redirect(url_for('records'))
        
        # 处理掩膜图片上传
        mask_image = request.files['mask_image']
        if mask_image and allowed_file(mask_image.filename):
            mask_filename = secure_filename(f"{username}_{int(datetime.now().timestamp())}_mask.{mask_image.filename.rsplit('.', 1)[1].lower()}")
            mask_image_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            mask_image.save(mask_image_path)
            mask_image_rel_path = f"uploads/{mask_filename}"
        else:
            flash('掩膜图片格式不正确')
            return redirect(url_for('records'))
        
        # 保存记录到数据库
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO records (username, initial_image, mask_image, detection_date, created_at) VALUES (?, ?, ?, ?, ?)',
                 (username, initial_image_rel_path, mask_image_rel_path, detection_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        
        flash('记录添加成功')
    except Exception as e:
        flash(f'添加记录失败: {str(e)}')
    
    return redirect(url_for('records'))

# 删除治疗记录
@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        username = session.get('username')
        
        # 获取记录信息，以便删除相关文件
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT initial_image, mask_image FROM records WHERE id = ? AND username = ?', (record_id, username))
        record = c.fetchone()
        
        if not record:
            return jsonify({'success': False, 'message': '记录不存在或无权限删除'})
        
        # 删除图片文件
        try:
            initial_image_path = os.path.join('static', record[0])
            if os.path.exists(initial_image_path):
                os.remove(initial_image_path)
                
            mask_image_path = os.path.join('static', record[1])
            if os.path.exists(mask_image_path):
                os.remove(mask_image_path)
        except Exception as e:
            print(f"删除文件错误: {str(e)}")
        
        # 从数据库中删除记录
        c.execute('DELETE FROM records WHERE id = ? AND username = ?', (record_id, username))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 批量删除治疗记录
@app.route('/delete_selected_records', methods=['POST'])
def delete_selected_records():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        username = session.get('username')
        record_ids = request.json.get('record_ids', [])
        
        if not record_ids:
            return jsonify({'success': False, 'message': '未选择任何记录'})
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # 获取所有要删除的记录信息，以便删除相关文件
        placeholders = ','.join(['?' for _ in record_ids])
        c.execute(f'SELECT id, initial_image, mask_image FROM records WHERE id IN ({placeholders}) AND username = ?', 
                 record_ids + [username])
        records = c.fetchall()
        
        if not records:
            return jsonify({'success': False, 'message': '记录不存在或无权限删除'})
        
        # 删除图片文件
        deleted_count = 0
        for record in records:
            try:
                initial_image_path = os.path.join('static', record[1])
                if os.path.exists(initial_image_path):
                    os.remove(initial_image_path)
                    
                mask_image_path = os.path.join('static', record[2])
                if os.path.exists(mask_image_path):
                    os.remove(mask_image_path)
                    
                # 从数据库中删除记录
                c.execute('DELETE FROM records WHERE id = ? AND username = ?', (record[0], username))
                deleted_count += 1
            except Exception as e:
                print(f"删除记录 {record[0]} 错误: {str(e)}")
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'成功删除 {deleted_count} 条记录'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 搜索治疗记录
@app.route('/search_records', methods=['GET'])
def search_records():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        username = session.get('username')
        search_term = request.args.get('q', '').strip()
        
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # 如果搜索词为空，返回所有记录
        if not search_term:
            c.execute('SELECT * FROM records WHERE username = ? ORDER BY detection_date DESC', (username,))
        else:
            # 搜索检测日期或创建时间包含搜索词的记录
            search_pattern = f'%{search_term}%'
            c.execute('SELECT * FROM records WHERE username = ? AND (detection_date LIKE ? OR created_at LIKE ?) ORDER BY detection_date DESC', 
                     (username, search_pattern, search_pattern))
        
        records = [dict(row) for row in c.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 获取记录详情
@app.route('/get_record/<int:record_id>', methods=['GET'])
def get_record(record_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': '未登录'})
    
    try:
        username = session.get('username')
        
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM records WHERE id = ? AND username = ?', (record_id, username))
        record = c.fetchone()
        conn.close()
        
        if not record:
            return jsonify({'success': False, 'message': '记录不存在或无权限查看'})
        
        return jsonify({'success': True, 'record': dict(record)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import os
import sqlite3
import hashlib
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
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
AVATARS_FOLDER = 'static/avatars'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['AVATARS_FOLDER'] = AVATARS_FOLDER

# 确保上传和结果目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(AVATARS_FOLDER, exist_ok=True)

# 确保默认头像文件存在 - 此处仅检查是否存在，具体创建在init_app()中完成
default_avatar_path = os.path.join(AVATARS_FOLDER, 'default-avatar.png')
if not os.path.exists(default_avatar_path):
    try:
        # 创建一个临时的默认头像
        from PIL import Image, ImageDraw
        
        # 使用灰色背景和蓝边框创建默认头像
        img = Image.new('RGB', (100, 100), color=(230, 230, 230))  # 浅灰色背景
        d = ImageDraw.Draw(img)
        # 绘制蓝色圆形
        d.ellipse((0, 0, 99, 99), fill=(65, 105, 225))  # 蓝色填充
        # 在中间绘制一个白色小圆形表示人物
        d.ellipse((25, 15, 75, 65), fill=(255, 255, 255))  # 白色头部
        # 绘制身体部分
        d.rectangle((40, 65, 60, 85), fill=(255, 255, 255))  # 白色身体
        
        img.save(default_avatar_path)
        print(f"初始创建默认头像文件: {default_avatar_path}")
    except Exception as e:
        print(f"创建默认头像文件失败: {str(e)}")
        # 创建一个回退方案，至少确保文件存在
        try:
            with open(default_avatar_path, 'wb') as f:
                # 写入一个简单的PNG数据
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82')
            print(f"创建小的默认头像文件: {default_avatar_path}")
        except Exception as e2:
            print(f"创建空默认头像文件也失败: {str(e2)}")

# 加载YOLOv11模型
detection_model = None
segmentation_model = None
classification_model = None

# 星火大模型配置
SPARK_APP_ID = 'b1968706'
SPARK_API_KEY = 'e6a007a32da370f149c75b34d10cdd74'
SPARK_API_SECRET = 'NTFmZGFlMThkMTQyNWZjZDcyZDAxOThm'

# 快速分析建议模板
QUICK_ANALYSIS_TEMPLATES = {
    "胶质瘤": {
        "description": "胶质瘤是最常见的原发性脑肿瘤，起源于神经胶质细胞。",
        "risk_levels": {
            "high": "高度恶性胶质瘤(WHO III-IV级)进展迅速，预后较差，需要积极治疗。",
            "medium": "中度恶性胶质瘤(WHO II级)生长较慢，但有恶变风险，需要定期随访。",
            "low": "低度恶性胶质瘤(WHO I级)生长缓慢，预后较好，但仍需定期随访。"
        },
        "follow_up": "建议每3-6个月进行一次MRI检查，监测肿瘤大小和特征变化。",
        "tips": "保持良好生活习惯，避免过度疲劳，定期随访，遵医嘱服药。"
    },
    "脑膜瘤": {
        "description": "脑膜瘤起源于脑膜，多为良性，生长缓慢，边界清晰。",
        "risk_levels": {
            "high": "少数恶性脑膜瘤(WHO II-III级)生长较快，可能侵袭周围组织，需要积极治疗。",
            "medium": "非典型脑膜瘤(WHO II级)复发风险增加，需要更频繁随访。",
            "low": "典型脑膜瘤(WHO I级)通常为良性，生长缓慢，预后良好。"
        },
        "follow_up": "建议每6-12个月进行一次MRI检查，监测肿瘤大小变化。",
        "tips": "避免剧烈运动，保持充足睡眠，定期随访，注意头痛、视力变化等症状。"
    },
    "垂体瘤": {
        "description": "垂体瘤是发生在垂体的肿瘤，可能影响激素分泌，导致内分泌紊乱。",
        "risk_levels": {
            "high": "大型垂体瘤(>4cm)可能压迫视神经和周围结构，需要积极治疗。",
            "medium": "中型垂体瘤(1-4cm)需要评估激素水平和视力影响，可能需要治疗。",
            "low": "微型垂体瘤(<1cm)通常可以观察随访，除非有明显症状或激素异常。"
        },
        "follow_up": "建议每6-12个月进行一次MRI检查和内分泌功能评估。",
        "tips": "定期检查激素水平，注意视力变化，保持健康生活方式，避免过度疲劳。"
    },
    "无脑肿瘤": {
        "description": "未检测到明确的脑肿瘤特征。",
        "risk_levels": {
            "high": "虽未检测到肿瘤，但如有持续症状，建议进一步检查。",
            "medium": "可能存在其他非肿瘤性病变，建议咨询神经科医生。",
            "low": "目前未见异常，建议保持健康生活习惯。"
        },
        "follow_up": "如无特殊症状，建议每年进行一次常规体检。",
        "tips": "保持健康生活方式，避免过度疲劳，如出现持续头痛、视力变化等症状应及时就医。"
    }
}

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
# load_models()  # 注释掉这行，因为我们已经在init_app()中调用了load_models()

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
            password TEXT NOT NULL,
            avatar TEXT DEFAULT 'default-avatar.png',
            bio TEXT DEFAULT '这个人很懒，什么都没有留下。'
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
    """检查上传的文件类型是否在允许列表中"""
    if '.' not in filename:
        print(f"文件名 {filename} 没有扩展名")
        return False
    
    file_ext = filename.rsplit('.', 1)[1].lower()
    is_allowed = file_ext in ALLOWED_EXTENSIONS
    
    if not is_allowed:
        print(f"文件扩展名 {file_ext} 不在允许列表中: {ALLOWED_EXTENSIONS}")
    
    return is_allowed

# 确保会话中有最新的用户头像
def ensure_avatar_in_session():
    if not session.get('logged_in') or not session.get('username'):
        return
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT avatar FROM users WHERE username = ?', (session['username'],))
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            # 只有当数据库中的头像与会话中的不同时才更新
            if 'avatar' not in session or session['avatar'] != result[0]:
                print(f"更新会话中的头像: {session.get('avatar', '无')} -> {result[0]}")
                session['avatar'] = result[0]
        else:
            if 'avatar' not in session:
                session['avatar'] = 'default-avatar.png'
    except Exception as e:
        print(f"获取用户头像失败: {str(e)}")
        if 'avatar' not in session:
            session['avatar'] = 'default-avatar.png'

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
            session['user_id'] = user[0]
            
            # 安全地获取头像和个性签名，如果不存在则使用默认值
            try:
                session['avatar'] = user[3] if len(user) > 3 and user[3] else 'default-avatar.png'
            except (IndexError, TypeError):
                session['avatar'] = 'default-avatar.png'
                
            try:
                session['bio'] = user[4] if len(user) > 4 and user[4] else '这个人很懒，什么都没有留下。'
            except (IndexError, TypeError):
                session['bio'] = '这个人很懒，什么都没有留下。'
                
            # 尝试更新用户表结构，确保有avatar和bio字段
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                # 检查users表是否有avatar列
                c.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in c.fetchall()]
                
                if 'avatar' not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN avatar TEXT DEFAULT 'default-avatar.png'")
                
                if 'bio' not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT '这个人很懒，什么都没有留下。'")
                
                # 确保当前用户有这些字段的值
                c.execute("UPDATE users SET avatar = ?, bio = ? WHERE username = ? AND (avatar IS NULL OR bio IS NULL)",
                         ('default-avatar.png', '这个人很懒，什么都没有留下。', username))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"更新用户表结构时出错: {str(e)}")
            
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
    
    # 处理头像选择
    avatar_filename = 'default-avatar.png'
    avatar_saved = False  # 标记是否成功保存了头像文件

    # 确保头像目录存在
    os.makedirs(app.config['AVATARS_FOLDER'], exist_ok=True)
    
    print("===== 开始处理注册头像 =====")
    print(f"表单数据: {request.form}")
    print(f"文件数据: {request.files}")
    # 先尝试处理自定义头像上传 - 优先级最高
    if 'avatar' in request.files:
        avatar_file = request.files['avatar']
        print(f"DEBUG: 注册时上传的头像文件: {avatar_file.filename if avatar_file else None}")
        
        if avatar_file and avatar_file.filename != '' and allowed_file(avatar_file.filename):
            try:
                # 生成安全的文件名
                filename = secure_filename(avatar_file.filename)
                # 添加时间戳避免文件名冲突
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                avatar_filename = f"{username}_{timestamp}_{filename}"
                avatar_path = os.path.join(app.config['AVATARS_FOLDER'], avatar_filename)
                avatar_file.save(avatar_path)
                
                # 检查文件是否成功保存并且非空
                if os.path.exists(avatar_path) and os.path.getsize(avatar_path) > 0:
                    print(f"成功保存自定义头像: {avatar_filename}，大小: {os.path.getsize(avatar_path)} 字节")
                    avatar_saved = True
                else:
                    print(f"自定义头像保存失败或文件为空: {avatar_path}")
                    avatar_filename = 'default-avatar.png'  # 回退到默认头像
            except Exception as e:
                print(f"保存自定义头像时出错: {str(e)}")
                avatar_filename = 'default-avatar.png'  # 出错时使用默认头像
    
    # 如果自定义头像处理失败，再检查是否选择了系统预设头像
    if not avatar_saved:
        # 检查是否选择了系统预设头像
        selected_system_avatar = request.form.get('selected_system_avatar')
        print(f"DEBUG: 注册时选择的系统头像: {selected_system_avatar}")
        
        if selected_system_avatar and selected_system_avatar != '':
            # 检查是否是完整的文件名或者只是doctor-1这样的标识符
            if selected_system_avatar in ['doctor-1', 'doctor-2', 'doctor-3']:
                # 找到对应的实际文件名
                if selected_system_avatar == 'doctor-1':
                    avatar_filename = 'b4265bcc71b199c3223c8090a8bda350.png'
                elif selected_system_avatar == 'doctor-2':
                    avatar_filename = '5c5c70487e68968ccb5fa2cdb4bde9a7.png'
                elif selected_system_avatar == 'doctor-3':
                    avatar_filename = '342cbbb4-2f57-40b7-8b21-a29766f80fc3.png'
            else:
                # 如果是完整文件名，直接使用
                avatar_filename = selected_system_avatar
            
            print(f"使用系统头像: {avatar_filename}")
            # 确保头像文件存在
            avatar_path = os.path.join(app.config['AVATARS_FOLDER'], avatar_filename)
            if not os.path.exists(avatar_path):
                print(f"警告: 选择的头像文件不存在: {avatar_path}")
                
                # 对于系统预设头像，如果不存在则创建一个空白的默认头像
                if avatar_filename in ['b4265bcc71b199c3223c8090a8bda350.png', '5c5c70487e68968ccb5fa2cdb4bde9a7.png', '342cbbb4-2f57-40b7-8b21-a29766f80fc3.png']:
                    try:
                        avatars_dir = app.config['AVATARS_FOLDER']
                        os.makedirs(avatars_dir, exist_ok=True)
                        
                        # 根据文件名选择不同的颜色
                        if avatar_filename == 'b4265bcc71b199c3223c8090a8bda350.png':
                            color = (200, 0, 0)  # 红色系
                        elif avatar_filename == '5c5c70487e68968ccb5fa2cdb4bde9a7.png':
                            color = (0, 0, 200)  # 蓝色系
                        else:
                            color = (0, 200, 0)  # 绿色系
                        
                        # 创建一个简单的颜色头像
                        from PIL import Image, ImageDraw
                        img = Image.new('RGB', (100, 100), color=color)
                        d = ImageDraw.Draw(img)
                        d.ellipse((10, 10, 90, 90), fill=(255, 255, 255))
                        img.save(avatar_path)
                        print(f"创建了缺失的系统头像: {avatar_path}")
                    except Exception as e:
                        print(f"创建系统头像失败 {avatar_filename}: {str(e)}")
                        avatar_filename = 'default-avatar.png'  # 失败时使用默认头像
                else:
                    avatar_saved = True
    
    print(f"DEBUG: 最终使用的头像文件名: {avatar_filename}")
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # 检查users表结构
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        
        # 确保表有必要的列
        if 'avatar' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN avatar TEXT DEFAULT 'default-avatar.png'")
        
        if 'bio' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT '这个人很懒，什么都没有留下。'")
        
        # 根据表结构插入数据
        print(f"DEBUG: 准备向数据库插入头像: {avatar_filename}")
        if 'avatar' in columns and 'bio' in columns:
            c.execute('INSERT INTO users (username, password, avatar, bio) VALUES (?, ?, ?, ?)', 
                     (username, hashed_password, avatar_filename, '这个人很懒，什么都没有留下。'))
        else:
            # 如果表结构仍有问题，只插入基本信息
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                     (username, hashed_password))
        
        conn.commit()
        
        # 在注册成功后自动登录
        # 获取新注册用户的信息，包括ID和头像
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        
        # 验证头像是否正确写入
        if user:
            print(f"DEBUG: 注册成功后从数据库获取的用户信息: {user}")
            print(f"DEBUG: 注册成功后从数据库获取的用户头像: {user[3] if len(user) > 3 else None}")
        
        conn.close()
        
        if user:
            # 设置会话信息
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = user[0]
            
            # 安全地获取头像和个性签名
            try:
                session['avatar'] = user[3] if len(user) > 3 and user[3] else 'default-avatar.png'
                print(f"DEBUG: 设置会话头像为: {session['avatar']}")
            except (IndexError, TypeError):
                session['avatar'] = 'default-avatar.png'
                print(f"DEBUG: 设置会话默认头像")
                
            try:
                session['bio'] = user[4] if len(user) > 4 and user[4] else '这个人很懒，什么都没有留下。'
            except (IndexError, TypeError):
                session['bio'] = '这个人很懒，什么都没有留下。'
                
            flash('注册成功，已自动登录')
            return redirect(url_for('diagnosis'))
        else:
            flash('注册成功，请登录')
    except sqlite3.IntegrityError:
        flash('用户名已存在')
    except Exception as e:
        flash(f'注册失败，请稍后重试: {str(e)}')
        print(f"DEBUG: 注册过程中发生异常: {str(e)}")
    
    return redirect(url_for('login'))

# 登出路由
@app.route('/logout')
def logout():
    # 清除所有会话信息
    session.clear()
    return redirect(url_for('login'))

# 肿瘤诊断页面
@app.route('/diagnosis')
def diagnosis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # 确保会话中有最新的用户头像
    ensure_avatar_in_session()
    
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
    
    # 确保会话中有最新的用户头像
    ensure_avatar_in_session()
    
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
        gaussian_blur = int(data.get('gaussian_blur', 0))  # 高斯滤波核大小
        contrast = float(data.get('contrast', 100))  # 对比度调整值
        image_path = data.get('image_path')
        
        print(f"接收到的图像路径: {image_path}")
        
        # 确保路径格式正确
        if image_path.startswith('/static/'):
            file_path = image_path.lstrip('/')
        else:
            file_path = image_path.lstrip('/')
            if not file_path.startswith('static/'):
                file_path = f"static/{file_path}"
        
        print(f"处理后的文件路径: {file_path}")
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': f'图像不存在: {file_path}'})
        
        # 重新处理图像，使用更敏感的阈值范围
        threshold_value = max(0.05, min(0.95, threshold / 100.0))  # 将阈值映射到0.05-0.95的范围
        
        # 读取原始图像
        original_img = cv2.imread(file_path)
        if original_img is None:
            return jsonify({'success': False, 'message': f'无法读取图像: {file_path}'})
            
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
        if model_type not in ['deepseek', 'qwen', 'quick']:
            model_type = 'deepseek'

        if output['detections']:
            # 计算良恶性概率
            malignant_probs = [m['probability'] for m in output['malignancy'] if m['class'] == '恶性']
            benign_probs = [m['probability'] for m in output['malignancy'] if m['class'] == '良性']

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
                if model_type == 'quick':
                    # 使用快速分析功能
                    output['medical_analysis'] = get_quick_analysis(
                        class_info=analysis_data['class_info'],
                        detection_info=analysis_data['detection_info']
                    )
                else:
                    # 使用大模型分析
                    output['medical_analysis'] = get_ai_response(
                        model_type=model_type,
                        class_info=analysis_data['class_info'],
                        detection_info=analysis_data['detection_info']
                    )
            except Exception as e:
                print(f"AI分析失败: {str(e)}")
                output['medical_analysis'] = f"医学分析生成失败，请稍后再试。错误信息：{str(e)}"

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
    return render_template('classification.html')

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
    
    # 确保会话中有最新的用户头像
    ensure_avatar_in_session()
    
    username = session.get('username')
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT id, initial_image, mask_image, detection_date, created_at
                FROM records WHERE username = ? ORDER BY created_at DESC''', (username,))
    records_data = c.fetchall()
    conn.close()
    
    records = []
    for record in records_data:
        records.append({
            'id': record[0],
            'initial_image': record[1],
            'mask_image': record[2],
            'detection_date': record[3],
            'created_at': record[4]
        })
    
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

# 快速分析功能
def get_quick_analysis(class_info, detection_info):
    """基于规则的快速医学建议分析"""
    tumor_type = class_info['class_name']
    probability = class_info['probability']
    malignant_prob = detection_info['malignant_prob']
    benign_prob = detection_info['benign_prob']
    
    # 获取对应肿瘤类型的模板
    template = QUICK_ANALYSIS_TEMPLATES.get(tumor_type, QUICK_ANALYSIS_TEMPLATES["无脑肿瘤"])
    
    # 确定风险等级
    risk_level = "low"
    if malignant_prob > 70:
        risk_level = "high"
    elif malignant_prob > 30:
        risk_level = "medium"
    
    # 生成分析结果
    analysis = f"""【快速医学分析】

▶ 肿瘤类型：{tumor_type}（置信度：{probability:.1f}%）
{template['description']}

▶ 风险评估：
{template['risk_levels'][risk_level]}
良性概率：{benign_prob:.1f}%，恶性概率：{malignant_prob:.1f}%

▶ 随访建议：
{template['follow_up']}

▶ 日常注意事项：
{template['tips']}

※ 注意：此分析仅供参考，请咨询专业医生获取准确诊断和治疗方案。"""
    
    return analysis

# 用户个人资料页面
@app.route('/profile')
def profile():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # 确保会话中有最新的用户头像
    ensure_avatar_in_session()
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (session['username'],))
    user = c.fetchone()
    conn.close()
    
    if not user:
        flash('用户信息获取失败')
        return redirect(url_for('diagnosis'))
    
    # 安全地构建用户数据字典
    user_data = {
        'id': user[0],
        'username': user[1],
        'avatar': user[3] if len(user) > 3 and user[3] else 'default-avatar.png',
        'bio': user[4] if len(user) > 4 and user[4] else '这个人很懒，什么都没有留下。'
    }
    
    # 更新会话中的用户信息
    session['avatar'] = user_data['avatar']
    session['bio'] = user_data['bio']
    
    return render_template('profile.html', user=user_data)

# 更新用户个人资料
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # 获取表单类型，根据不同的表单处理不同的更新操作
    form_type = request.form.get('form_type', 'basic_info')
    
    # 处理基本资料更新（头像和个性签名）
    if form_type == 'basic_info':
        bio = request.form.get('bio', '这个人很懒，什么都没有留下。')
        
        # 处理头像上传
        avatar_filename = session.get('avatar', 'default-avatar.png')
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file and avatar_file.filename != '' and allowed_file(avatar_file.filename):
                # 生成安全的文件名
                filename = secure_filename(avatar_file.filename)
                # 添加时间戳避免文件名冲突
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                avatar_filename = f"{session['username']}_{timestamp}_{filename}"
                avatar_file.save(os.path.join(app.config['AVATARS_FOLDER'], avatar_filename))
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # 更新用户资料
            c.execute('UPDATE users SET avatar = ?, bio = ? WHERE username = ?', 
                     (avatar_filename, bio, session['username']))
            conn.commit()
            conn.close()
            
            # 更新会话中的用户信息
            session['avatar'] = avatar_filename
            session['bio'] = bio
            
            flash('个人资料更新成功')
        except Exception as e:
            flash(f'个人资料更新失败，请稍后重试: {str(e)}')
    
    # 处理账号设置更新（用户名）
    elif form_type == 'account_settings':
        new_username = request.form.get('username')
        current_password = request.form.get('current_password')
        
        if not new_username or not current_password:
            flash('用户名和当前密码不能为空')
            return redirect(url_for('profile'))
        
        # 验证当前密码是否正确
        hashed_password = hashlib.sha256(current_password.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # 验证密码
            c.execute('SELECT id FROM users WHERE username = ? AND password = ?', 
                     (session['username'], hashed_password))
            user = c.fetchone()
            
            if not user:
                flash('当前密码不正确')
                conn.close()
                return redirect(url_for('profile'))
            
            # 检查新用户名是否已存在（如果新用户名与当前用户名不同）
            if new_username != session['username']:
                c.execute('SELECT id FROM users WHERE username = ?', (new_username,))
                existing_user = c.fetchone()
                
                if existing_user:
                    flash('用户名已被占用，请选择其他用户名')
                    conn.close()
                    return redirect(url_for('profile'))
                
                # 更新用户名
                c.execute('UPDATE users SET username = ? WHERE id = ?', 
                         (new_username, user[0]))
                
                # 更新关联记录中的用户名（如果有记录表）
                try:
                    c.execute('UPDATE records SET username = ? WHERE username = ?', 
                             (new_username, session['username']))
                except Exception as e:
                    print(f"更新记录表用户名时出错: {str(e)}")
                
                conn.commit()
                
                # 更新会话
                old_username = session['username']
                session['username'] = new_username
                
                # 如果头像名称中包含用户名，可能需要更新头像文件名
                if session.get('avatar') and old_username in session['avatar']:
                    try:
                        old_avatar = session['avatar']
                        new_avatar = old_avatar.replace(old_username, new_username)
                        old_path = os.path.join(app.config['AVATARS_FOLDER'], old_avatar)
                        new_path = os.path.join(app.config['AVATARS_FOLDER'], new_avatar)
                        
                        if os.path.exists(old_path):
                            # 复制文件到新名称
                            shutil.copy2(old_path, new_path)
                            
                            # 更新数据库中的头像名称
                            c.execute('UPDATE users SET avatar = ? WHERE id = ?', 
                                     (new_avatar, user[0]))
                            conn.commit()
                            
                            # 更新会话中的头像
                            session['avatar'] = new_avatar
                    except Exception as e:
                        print(f"更新头像文件名时出错: {str(e)}")
                
                flash('用户名更新成功')
            else:
                flash('用户名保持不变')
            
            conn.close()
        except Exception as e:
            flash(f'更新用户名失败，请稍后重试: {str(e)}')
    
    # 处理密码更新
    elif form_type == 'password_change':
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not old_password or not new_password or not confirm_password:
            flash('所有密码字段都不能为空')
            return redirect(url_for('profile'))
        
        if new_password != confirm_password:
            flash('新密码和确认密码不匹配')
            return redirect(url_for('profile'))
        
        # 验证旧密码
        hashed_old_password = hashlib.sha256(old_password.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            c.execute('SELECT id FROM users WHERE username = ? AND password = ?', 
                     (session['username'], hashed_old_password))
            user = c.fetchone()
            
            if not user:
                flash('当前密码不正确')
                conn.close()
                return redirect(url_for('profile'))
            
            # 对新密码进行哈希处理
            hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
            
            # 更新密码
            c.execute('UPDATE users SET password = ? WHERE id = ?', 
                     (hashed_new_password, user[0]))
            conn.commit()
            conn.close()
            
            flash('密码更新成功')
        except Exception as e:
            flash(f'更新密码失败，请稍后重试: {str(e)}')
    
    # 确保静态文件夹存在
    avatars_dir = os.path.join('static', 'avatars')
    if not os.path.exists(avatars_dir):
        os.makedirs(avatars_dir)

    # 如果是默认头像，确保默认头像文件存在
    if session.get('avatar') == 'default-avatar.png':
        default_avatar_path = os.path.join(avatars_dir, 'default-avatar.png')
        try:
            # 如果默认头像不存在或为空，强制创建新的默认头像
            if not os.path.exists(default_avatar_path) or os.path.getsize(default_avatar_path) == 0:
                # 创建一个简单的默认头像
                from PIL import Image, ImageDraw
                # 使用灰色背景和蓝边框创建默认头像
                img = Image.new('RGB', (100, 100), color=(230, 230, 230))  # 浅灰色背景
                d = ImageDraw.Draw(img)
                # 绘制蓝色圆形
                d.ellipse((0, 0, 99, 99), fill=(65, 105, 225))  # 蓝色填充
                # 在中间绘制一个白色小圆形表示人物
                d.ellipse((25, 15, 75, 65), fill=(255, 255, 255))  # 白色头部
                # 绘制身体部分
                d.rectangle((40, 65, 60, 85), fill=(255, 255, 255))  # 白色身体
                
                img.save(default_avatar_path)
                print(f"在个人资料更新时创建了新的默认头像: {default_avatar_path}")
        except Exception as e:
            print(f"创建默认头像失败: {str(e)}")
        # 如果PIL创建失败，至少确保文件存在
        try:
            if not os.path.exists(default_avatar_path) or os.path.getsize(default_avatar_path) == 0:
                with open(default_avatar_path, 'wb') as f:
                    # 写入一个简单的PNG数据
                    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82')
                print(f"创建了小的默认头像文件")
        except Exception as e2:
            print(f"创建默认头像文件也失败: {str(e2)}")
    
    return redirect(url_for('profile'))

# 获取用户头像
@app.route('/avatars/<filename>')
def get_avatar(filename):
    try:
        # 尝试发送请求的头像文件
        avatar_path = os.path.join(app.config['AVATARS_FOLDER'], filename)
        print(f"请求头像: {filename}, 完整路径: {avatar_path}")
        
        # 检查文件是否存在
        if os.path.exists(avatar_path):
            file_size = os.path.getsize(avatar_path)
            print(f"头像文件存在，大小: {file_size} 字节")
            
            if file_size > 0:
                print(f"返回请求的头像: {filename}")
                return send_from_directory(app.config['AVATARS_FOLDER'], filename)
            else:
                print(f"头像文件为空: {filename}")
        else:
            print(f"头像文件不存在: {filename}")
        
        # 如果执行到这里，说明需要返回默认头像
        default_avatar = 'default-avatar.png'
        default_path = os.path.join(app.config['AVATARS_FOLDER'], default_avatar)
        
        # 检查默认头像是否存在
        if not os.path.exists(default_path) or os.path.getsize(default_path) == 0:
            # 如果默认头像不存在或为空，创建一个新的
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color=(65, 105, 225))  # 蓝色背景
                d = ImageDraw.Draw(img)
                # 绘制白色圆形
                d.ellipse((25, 15, 75, 65), fill=(255, 255, 255))  # 白色头部
                # 绘制身体部分
                d.rectangle((40, 65, 60, 85), fill=(255, 255, 255))  # 白色身体
                
                img.save(default_path)
                print(f"在get_avatar中创建了新的默认头像: {default_path}")
            except Exception as e:
                print(f"创建默认头像失败: {str(e)}")
        
        print(f"返回默认头像: {default_avatar}")
        return send_from_directory(app.config['AVATARS_FOLDER'], default_avatar)
    except Exception as e:
        print(f"头像访问错误: {str(e)}")
        # 发生严重错误时，返回一个内联的SVG头像
        svg_avatar = '''
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="50" fill="#4169E1"/>
            <circle cx="50" cy="40" r="25" fill="#FFFFFF"/>
            <rect x="40" y="65" width="20" height="20" fill="#FFFFFF"/>
        </svg>
        '''
        response = make_response(svg_avatar)
        response.headers['Content-Type'] = 'image/svg+xml'
        return response

# 初始化应用
def init_app():
    # 初始化数据库
    init_db()
    
    # 加载模型
    load_models()
    
    # 检查并修复数据库结构
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # 检查users表结构
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        
        # 确保表有必要的列
        table_modified = False
        if 'avatar' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN avatar TEXT DEFAULT 'default-avatar.png'")
            table_modified = True
            print("添加avatar列到users表")
        
        if 'bio' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT '这个人很懒，什么都没有留下。'")
            table_modified = True
            print("添加bio列到users表")
        
        if table_modified:
            conn.commit()
            print("数据库结构已更新")
        
        # 更新现有用户的空值
        c.execute("UPDATE users SET avatar = 'default-avatar.png' WHERE avatar IS NULL")
        c.execute("UPDATE users SET bio = '这个人很懒，什么都没有留下。' WHERE bio IS NULL")
        conn.commit()
        
        conn.close()
    except Exception as e:
        print(f"检查数据库结构时出错: {str(e)}")
    
    # 确保静态文件夹存在
    avatars_dir = os.path.join('static', 'avatars')
    if not os.path.exists(avatars_dir):
        os.makedirs(avatars_dir)
        print(f"创建头像文件夹: {avatars_dir}")
    
    # 强制更新默认头像文件
    default_avatar_path = os.path.join(avatars_dir, 'default-avatar.png')
    try:
        # 删除现有的默认头像
        if os.path.exists(default_avatar_path):
            os.remove(default_avatar_path)
            print(f"删除现有的默认头像文件: {default_avatar_path}")
        
        # 创建一个简单的默认头像
        from PIL import Image, ImageDraw
        # 使用灰色背景和蓝边框创建默认头像
        img = Image.new('RGB', (100, 100), color=(230, 230, 230))  # 浅灰色背景
        d = ImageDraw.Draw(img)
        # 绘制蓝色圆形
        d.ellipse((0, 0, 99, 99), fill=(65, 105, 225))  # 蓝色填充
        # 在中间绘制一个白色小圆形表示人物
        d.ellipse((25, 15, 75, 65), fill=(255, 255, 255))  # 白色头部
        # 绘制身体部分
        d.rectangle((40, 65, 60, 85), fill=(255, 255, 255))  # 白色身体
        
        img.save(default_avatar_path)
        print(f"更新了默认头像: {default_avatar_path}")
    except Exception as e:
        print(f"创建默认头像失败: {str(e)}")
        # 如果PIL创建失败，至少创建一个空文件
        with open(default_avatar_path, 'w') as f:
            f.write('')
        print(f"创建默认头像失败: {str(e)}")
    
    # 确保系统预设头像存在
    system_avatars = {
        'b4265bcc71b199c3223c8090a8bda350.png': (200, 0, 0),  # 红色系
        '5c5c70487e68968ccb5fa2cdb4bde9a7.png': (0, 0, 200),   # 蓝色系
        '342cbbb4-2f57-40b7-8b21-a29766f80fc3.png': (0, 200, 0)  # 绿色系
    }
    
    for avatar_file, color in system_avatars.items():
        avatar_path = os.path.join(avatars_dir, avatar_file)
        # 只在文件不存在时才创建，不修改已存在的系统头像
        if not os.path.exists(avatar_path):
            try:
                # 如果系统预设头像不存在，创建一个简单的替代头像
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color=color)
                d = ImageDraw.Draw(img)
                d.ellipse((10, 10, 90, 90), fill=(255, 255, 255))
                img.save(avatar_path)
                print(f"创建了缺失的系统头像: {avatar_path}")
            except Exception as e:
                print(f"创建系统头像失败 {avatar_file}: {str(e)}")
        else:
            print(f"系统头像已存在，保持原样: {avatar_file}")

# 在应用启动时初始化
init_app()

if __name__ == '__main__':
    app.run(debug=True)
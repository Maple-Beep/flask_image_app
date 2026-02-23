# flask_image_app/app.py
# ========================================================================
# flask_image_app/app.py
# 主应用文件 - 已添加增强的AI报告生成功能
# ========================================================================
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, abort, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from config import Config
import random

# --- ✅ 使用增强版的推理引擎 ---
from inference_engine.engine import MedicalReportEngine

# --- 初始化Flask应用 ---
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "请先登录以访问此页面."


# --- 🔑 必须添加 user_loader 回调 ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- 数据库模型 (修改点：添加级联删除) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    # 修改此处：添加 cascade 参数，解决完整性错误
    images = db.relationship('ImageModel', backref='user', lazy=True, cascade="all, delete")


class ImageModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    original_filename = db.Column(db.String(120))  # 添加原始文件名字段
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.now)
    ai_report = db.Column(db.Text)


# --- 完整路由定义 ---
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面 - 支持用户名或邮箱登录"""
    if request.method == 'POST':
        identifier = request.form['username_or_email']
        password = request.form['password']
        user = User.query.filter_by(username=identifier).first()
        if not user:
            user = User.query.filter_by(email=identifier).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("登录成功！", "success")
            return redirect(url_for('user_profile'))
        else:
            flash("用户名/邮箱或密码错误。", "error")
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash("两次输入的密码不一致。", "error")
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash("用户名已存在。", "error")
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash("邮箱已存在。", "error")
            return render_template('register.html')
        new_user = User(username=username, email=email,
                        password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash("注册成功，请登录。", "success")
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """退出登录"""
    logout_user()
    flash("已退出登录。", "info")
    return redirect(url_for('index'))


@app.route('/user_profile')
@login_required
def user_profile():
    """用户个人中心"""
    user_images = ImageModel.query.filter_by(user_id=current_user.id).all()
    return render_template('user_profile.html', images=user_images)


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    """管理员登录 - 支持用户名或邮箱登录"""
    if request.method == 'POST':
        identifier = request.form['username_or_email']
        password = request.form['password']
        user = User.query.filter_by(username=identifier).first()
        if not user:
            user = User.query.filter_by(email=identifier).first()
        if user and check_password_hash(user.password_hash, password) and user.is_admin:
            login_user(user)
            flash("管理员登录成功！", "success")
            return redirect(url_for('admin_profile'))
        else:
            flash("管理员用户名/邮箱或密码错误。", "error")
    return render_template('admin_login.html')


@app.route('/admin_profile')
@login_required
def admin_profile():
    """管理员个人中心"""
    all_users = User.query.all()
    all_images = ImageModel.query.all()
    return render_template('admin_profile.html', users=all_users, images=all_images)


@app.route('/upload_image', methods=['GET', 'POST'])
@login_required
def upload_image():
    """上传图片 - 使用原始文件名（安全处理）"""
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename != '':
            original_filename = file.filename
            filename = secure_filename(file.filename)
            
            # 如果文件名已存在，添加时间戳
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            new_image = ImageModel(
                filename=filename, 
                original_filename=original_filename,
                user_id=current_user.id
            )
            db.session.add(new_image)
            db.session.commit()
            flash("图片上传成功！", "success")
            return redirect(url_for('user_profile'))
    return render_template('upload_image.html')


@app.route('/manage_users')
@login_required
def manage_users():
    """管理用户"""
    if not current_user.is_admin:
        abort(403)
    users = User.query.all()
    return render_template('manage_users.html', users=users)


@app.route('/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    """删除用户 - 包含数据库记录和磁盘文件"""
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(user_id)

    # 1. 先删除该用户的所有图片文件 (物理删除)
    for image in user.images:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # 2. 删除用户 (由于模型设置了级联, 图片记录会自动删除)
    db.session.delete(user)
    db.session.commit()

    flash("用户及其所有图片已成功删除。", "success")
    return redirect(url_for('manage_users'))


@app.route('/manage_images')
@login_required
def manage_images():
    """管理图片"""
    if not current_user.is_admin:
        abort(403)
    images = ImageModel.query.all()
    return render_template('manage_images.html', images=images)


@app.route('/delete_image/<int:image_id>')
@login_required
def delete_image(image_id):
    """删除图片"""
    if not current_user.is_admin:
        abort(403)
    image = ImageModel.query.get_or_404(image_id)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    db.session.delete(image)
    db.session.commit()
    flash("图片已删除。", "success")
    return redirect(url_for('manage_images'))


@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- 新增：忘记密码流程 ---
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """忘记密码"""
    if request.method == 'POST':
        if 'step' not in session:
            email = request.form['email']
            user = User.query.filter_by(email=email).first()
            if not user:
                flash("该邮箱未注册。", "error")
                return render_template('forgot_password.html', step=1)

            verification_code = str(random.randint(100000, 999999))
            session['reset_email'] = email
            session['verification_code'] = verification_code
            session['step'] = 2

            print(f"\n【密码重置】\n邮箱: {email}\n验证码: {verification_code}\n")

            flash("验证码已生成，请查看PyCharm终端。", "info")
            return render_template('forgot_password.html', step=2)

        elif session['step'] == 2:
            input_code = request.form['verification_code']
            if input_code == session['verification_code']:
                session['step'] = 3
                flash("验证码正确，请设置新密码。", "success")
                return render_template('forgot_password.html', step=3)
            else:
                flash("验证码错误，请重试。", "error")
                return render_template('forgot_password.html', step=2)

        elif session['step'] == 3:
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            if new_password != confirm_password:
                flash("两次输入的密码不一致。", "error")
                return render_template('forgot_password.html', step=3)

            email = session['reset_email']
            user = User.query.filter_by(email=email).first()
            if user:
                user.password_hash = generate_password_hash(new_password)
                db.session.commit()
                session.pop('reset_email', None)
                session.pop('verification_code', None)
                session.pop('step', None)
                flash("密码已成功重置，请登录。", "success")
                return redirect(url_for('login'))
            else:
                flash("系统错误，请重试。", "error")
                return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html', step=1)


# --- 增强的AI报告生成路由 ---
@app.route('/generate_report/<int:image_id>', methods=['POST'])
@login_required
def generate_report(image_id):
    """为指定图片生成AI报告（使用增强的采样策略）"""
    image = ImageModel.query.get_or_404(image_id)
    if image.user_id != current_user.id:
        abort(403)

    report_engine = app.report_engine

    if report_engine is None or report_engine.model is None:
        flash("AI报告服务当前不可用。", "error")
        return redirect(url_for('user_profile'))

    image_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], image.filename)

    if not os.path.exists(image_path):
        flash("图像文件不存在，无法生成报告。", "error")
        return redirect(url_for('user_profile'))

    report_text = report_engine.generate(
        image_path,
        temperature=0.8,   # 温度：0.8 较保守，质量稳定
        top_k=30,          # Top-K：30 平衡多样性与质量
        top_p=0.9,         # Top-P 核采样
    )
    
    image.ai_report = report_text
    db.session.commit()
    flash("AI报告已生成！", "success")
    return redirect(url_for('user_profile'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin_user = User(username='admin', email='admin@example.com',
                              password_hash=generate_password_hash('admin123'), is_admin=True)
            db.session.add(admin_user)
            db.session.commit()
            print("初始管理员账户已创建: 用户名 'admin', 密码 'admin123'")

        # --- 构建配置字典并初始化引擎（Transformer架构参数）---
        engine_config = {
            'MODEL_PATH': app.config['MODEL_PATH'],
            'VOCAB_PATH': app.config['VOCAB_PATH'],
            'IMG_SIZE': app.config['IMG_SIZE'],
            'IMG_MEAN': app.config['IMG_MEAN'],
            'IMG_STD': app.config['IMG_STD'],
            'VOCAB_SIZE': app.config['VOCAB_SIZE'],
            # Transformer 架构参数（对应训练工程）
            'D_MODEL': app.config['D_MODEL'],
            'NHEAD': app.config['NHEAD'],
            'NUM_LAYERS': app.config['NUM_LAYERS'],
            'DROPOUT': app.config['DROPOUT'],
            'MAX_REPORT_LEN': app.config['MAX_REPORT_LEN'],
            'PAD_TOKEN_ID': app.config['PAD_TOKEN_ID'],
            'SOS_TOKEN_ID': app.config['SOS_TOKEN_ID'],
            'EOS_TOKEN_ID': app.config['EOS_TOKEN_ID'],
        }

        # ✅ 初始化 Transformer 版 MedicalReportEngine
        app.report_engine = MedicalReportEngine(config_dict=engine_config, debug=False)

    app.run(debug=True)

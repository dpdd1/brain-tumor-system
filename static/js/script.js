// 登录页面密码显示/隐藏切换
document.addEventListener('DOMContentLoaded', function() {
    // 密码显示/隐藏切换功能
    const passwordToggles = document.querySelectorAll('.password-toggle');
    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const passwordInput = this.parentElement.querySelector('input');
            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                this.innerHTML = '<i class="fas fa-eye-slash"></i>';
            } else {
                passwordInput.type = 'password';
                this.innerHTML = '<i class="fas fa-eye"></i>';
            }
        });
    });
    
    // 登录/注册表单切换功能
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const switchToRegister = document.getElementById('switchToRegister');
    const switchToLogin = document.getElementById('switchToLogin');
    
    if (switchToRegister) {
        switchToRegister.addEventListener('click', function() {
            loginForm.style.display = 'none';
            registerForm.style.display = 'block';
            // 添加动画效果
            registerForm.classList.add('fade-in');
            setTimeout(() => {
                registerForm.classList.remove('fade-in');
            }, 1000);
        });
    }
    
    if (switchToLogin) {
        switchToLogin.addEventListener('click', function() {
            registerForm.style.display = 'none';
            loginForm.style.display = 'block';
            // 添加动画效果
            loginForm.classList.add('fade-in');
            setTimeout(() => {
                loginForm.classList.remove('fade-in');
            }, 1000);
        });
    }
    
    // 注册按钮点击事件
    const registerBtn = document.getElementById('registerBtn');
    if (registerBtn) {
        registerBtn.addEventListener('click', function() {
            const username = document.getElementById('reg-username').value;
            const password = document.getElementById('reg-password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            const messageDiv = document.getElementById('register-message');
            
            if (!username || !password || !confirmPassword) {
                messageDiv.textContent = '请填写所有必填字段';
                return;
            }
            
            if (password !== confirmPassword) {
                messageDiv.textContent = '两次输入的密码不一致';
                return;
            }
            
            // 创建表单数据并提交到注册路由
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            
            // 添加加载动画
            registerBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中...';
            registerBtn.disabled = true;
            
            fetch('/register', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(() => {
                // 注册后刷新页面，显示flash消息
                window.location.reload();
            })
            .catch(error => {
                console.error('注册错误:', error);
                messageDiv.textContent = '注册失败，请稍后重试';
                registerBtn.innerHTML = '注册 <i class="fas fa-user-plus"></i>';
                registerBtn.disabled = false;
            });
        });
    }
    
    // 添加输入框焦点效果
    const inputFields = document.querySelectorAll('input');
    inputFields.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('input-focus');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('input-focus');
        });
    });
    
    // 添加按钮悬停效果
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.classList.add('button-hover');
        });
        
        button.addEventListener('mouseleave', function() {
            this.classList.remove('button-hover');
        });
    });
});

// 图像预览功能
function previewImage(input, previewId) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function(e) {
            document.getElementById(previewId).src = e.target.result;
        }
        
        reader.readAsDataURL(input.files[0]);
    }
}

// 通用的模态框功能
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'flex';
        // 添加动画效果
        modal.classList.add('fade-in');
        setTimeout(() => {
            modal.classList.remove('fade-in');
        }, 500);
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        // 添加淡出动画
        modal.classList.add('fade-out');
        setTimeout(() => {
            modal.style.display = 'none';
            modal.classList.remove('fade-out');
        }, 300);
    }
}

// 关闭所有模态框
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        closeModal(event.target.id);
    }
};
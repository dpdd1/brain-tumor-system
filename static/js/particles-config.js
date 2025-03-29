// 粒子效果配置文件
document.addEventListener('DOMContentLoaded', function() {
    // 粒子效果配置并添加鼠标交互
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            "particles": {
                "number": {
                    "value": 40,  // 进一步减少粒子数量，从60降到40
                    "density": {
                        "enable": true,
                        "value_area": 1000  // 增加区域值，使粒子分布更加稀疏
                    }
                },
                "color": {
                    "value": ["#2196f3", "#64b5f6", "#bbdefb", "#4caf50", "#9c27b0"]
                },
                "shape": {
                    "type": ["circle", "triangle"],
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                    "polygon": {
                        "nb_sides": 5
                    }
                },
                "opacity": {
                    "value": 0.4,  // 进一步降低不透明度
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 1,
                        "opacity_min": 0.1,
                        "sync": false
                    }
                },
                "size": {
                    "value": 3,
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 5,
                        "size_min": 0.1,
                        "sync": false
                    }
                },
                "line_linked": {
                    "enable": true,
                    "distance": 250,  // 进一步增加连线的最小距离，减少线条数量
                    "color": "#2196f3",
                    "opacity": 0.3,  // 降低线条不透明度
                    "width": 1.2  // 减小线条宽度
                },
                "move": {
                    "enable": true,
                    "speed": 2,  // 降低移动速度
                    "direction": "none",
                    "random": true,
                    "straight": false,
                    "out_mode": "bounce",
                    "bounce": true,
                    "attract": {
                        "enable": true,
                        "rotateX": 600,
                        "rotateY": 1200
                    }
                }
            },
            "interactivity": {
                "detect_on": "window",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "grab"
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "connect"
                    },
                    "resize": true
                },
                "modes": {
                    "grab": {
                        "distance": 180,
                        "line_linked": {
                            "opacity": 0.5
                        }
                    },
                    "bubble": {
                        "distance": 200,
                        "size": 6,
                        "duration": 2,
                        "opacity": 0.8,
                        "speed": 3
                    },
                    "repulse": {
                        "distance": 200,
                        "duration": 0.4
                    },
                    "push": {
                        "particles_nb": 4
                    },
                    "remove": {
                        "particles_nb": 2
                    },
                    "connect": {
                        "distance": 250,
                        "line_linked": {
                            "opacity": 0.4
                        }
                    }
                }
            },
            "retina_detect": true
        });
    }

    // 鼠标移动效果
    document.addEventListener('mousemove', function(e) {
        // 鼠标坐标
        const mouseX = e.clientX;
        const mouseY = e.clientY;
        
        // 窗口尺寸
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        
        // 计算鼠标位置的相对值（-1到1之间）
        const relativeX = (mouseX / windowWidth) * 2 - 1;
        const relativeY = (mouseY / windowHeight) * 2 - 1;
        
        // 背景形状跟随鼠标移动
        document.querySelectorAll('.page-background .shape, .animated-background .shape').forEach(shape => {
            const shiftX = relativeX * 25; // 增加水平移动幅度
            const shiftY = relativeY * 25; // 增加垂直移动幅度
            
            // 应用轻微的位移
            shape.style.transform = `translate(${shiftX}px, ${shiftY}px)`;
        });
        
        // 注意：医学元素的鼠标跟随交互已被移除，元素将只保留CSS动画效果
    });
}); 
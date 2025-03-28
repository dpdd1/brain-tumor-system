// 自定义渐变颜色生成
const generateGradient = (ctx, chartArea, colorStart, colorEnd) => {
    const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
};

// 图表颜色主题 - 更新为更现代的配色方案
const chartColors = {
    purple: {
        primary: 'rgba(124, 77, 255, 1)',
        light: 'rgba(124, 77, 255, 0.2)',
        gradient: ['rgba(124, 77, 255, 0.0)', 'rgba(124, 77, 255, 0.5)']
    },
    blue: {
        primary: 'rgba(56, 129, 248, 1)',
        light: 'rgba(56, 129, 248, 0.2)',
        gradient: ['rgba(56, 129, 248, 0.0)', 'rgba(56, 129, 248, 0.5)']
    },
    teal: {
        primary: 'rgba(11, 186, 165, 1)',
        light: 'rgba(11, 186, 165, 0.2)',
        gradient: ['rgba(11, 186, 165, 0.0)', 'rgba(11, 186, 165, 0.5)']
    },
    pink: {
        primary: 'rgba(245, 54, 92, 1)',
        light: 'rgba(245, 54, 92, 0.2)',
        gradient: ['rgba(245, 54, 92, 0.0)', 'rgba(245, 54, 92, 0.5)']
    },
    amber: {
        primary: 'rgba(251, 99, 64, 1)',
        light: 'rgba(251, 99, 64, 0.2)',
        gradient: ['rgba(251, 99, 64, 0.0)', 'rgba(251, 99, 64, 0.5)']
    },
    green: {
        primary: 'rgba(45, 206, 137, 1)',
        light: 'rgba(45, 206, 137, 0.2)',
        gradient: ['rgba(45, 206, 137, 0.0)', 'rgba(45, 206, 137, 0.5)']
    }
};

// 饼图配色方案 - 更新为更协调的颜色搭配
const pieChartColors = [
    'rgba(94, 114, 228, 0.9)',
    'rgba(45, 206, 137, 0.9)',
    'rgba(251, 99, 64, 0.9)',
    'rgba(17, 205, 239, 0.9)',
    'rgba(245, 54, 92, 0.9)',
    'rgba(130, 94, 228, 0.9)',
    'rgba(45, 206, 183, 0.9)',
    'rgba(251, 140, 64, 0.9)',
    'rgba(17, 160, 239, 0.9)',
    'rgba(230, 126, 34, 0.9)'
];

// 图表阴影效果
const addShadowEffect = (chart) => {
    const ctx = chart.ctx;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    ctx.shadowBlur = 10;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 4;
};

// 根组件
const Dashboard = () => {
    const [data, setData] = React.useState(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);
    const [lastUpdated, setLastUpdated] = React.useState(null);
    // 添加轮播控制状态
    const [currentPage, setCurrentPage] = React.useState(0);
    const recordsPerPage = 5; // 每页显示5条记录

    // 图表引用
    const tumorChartRef = React.useRef(null);
    const userDetectionChartRef = React.useRef(null);
    const userConfidenceChartRef = React.useRef(null);
    const recentDetectionsChartRef = React.useRef(null);
    const detectionSourceChartRef = React.useRef(null);
    
    // 图表实例
    const tumorChartInstance = React.useRef(null);
    const userDetectionChartInstance = React.useRef(null);
    const userConfidenceChartInstance = React.useRef(null);
    const recentDetectionsChartInstance = React.useRef(null);
    const detectionSourceChartInstance = React.useRef(null);

    // 获取数据
    const fetchData = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/data');
            if (!response.ok) {
                throw new Error('服务器响应错误');
            }
            const result = await response.json();
            setData(result);
            setLastUpdated(new Date());
            setError(null);
        } catch (err) {
            console.error('获取数据失败:', err);
            setError('获取数据失败: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    // 初始加载数据
    React.useEffect(() => {
        fetchData();
        
        // 设置定时刷新（每30秒）
        const intervalId = setInterval(fetchData, 30000);
        
        // 清理函数
        return () => {
            clearInterval(intervalId);
            // 销毁所有图表实例
            if (tumorChartInstance.current) tumorChartInstance.current.destroy();
            if (userDetectionChartInstance.current) userDetectionChartInstance.current.destroy();
            if (userConfidenceChartInstance.current) userConfidenceChartInstance.current.destroy();
            if (recentDetectionsChartInstance.current) recentDetectionsChartInstance.current.destroy();
            if (detectionSourceChartInstance.current) detectionSourceChartInstance.current.destroy();
        };
    }, []);
    
    // 添加轮播定时器
    React.useEffect(() => {
        if (!data || !data.realtime_detections || data.realtime_detections.length <= recordsPerPage) {
            return;
        }
        
        const totalPages = Math.ceil(data.realtime_detections.length / recordsPerPage);
        const carouselInterval = setInterval(() => {
            setCurrentPage(prevPage => (prevPage + 1) % totalPages);
        }, 5000); // 每5秒切换一次
        
        return () => clearInterval(carouselInterval);
    }, [data]);

    // 更新图表函数
    const updateCharts = React.useCallback(() => {
        if (!data) return;
        
        // 更新肿瘤类型图表
        if (tumorChartRef.current) {
            const ctx = tumorChartRef.current.getContext('2d');
            if (tumorChartInstance.current) {
                tumorChartInstance.current.destroy();
            }
            
            // 固定的肿瘤类型标签
            const fixedLabels = ['脑膜瘤', '垂体瘤', '胶质瘤', '无脑肿瘤'];
            
            // 处理数据
            const tumorMap = {};
            data.tumor_types.forEach(item => {
                let type = item.tumor_type;
                if (type && type.startsWith('"') && type.endsWith('"')) {
                    type = type.substring(1, type.length - 1);
                }
                tumorMap[type || '未知'] = item.count;
            });
            
            // 根据固定标签获取对应值，如果没有则为0
            const counts = fixedLabels.map(label => {
                // 处理一些可能的匹配
                if (label === '脑膜瘤') {
                    return tumorMap['脑膜瘤'] || tumorMap['meningioma'] || 0;
                } else if (label === '垂体瘤') {
                    return tumorMap['垂体瘤'] || tumorMap['pituitary'] || tumorMap['pituitary tumor'] || 0;
                } else if (label === '胶质瘤') {
                    return tumorMap['胶质瘤'] || tumorMap['glioma'] || 0;
                } else if (label === '无脑肿瘤') {
                    return tumorMap['正常'] || tumorMap['no tumor'] || tumorMap['normal'] || tumorMap['无脑肿瘤'] || 0;
                }
                return 0;
            });
            
            // 创建图表
            tumorChartInstance.current = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: fixedLabels,
                    datasets: [{
                        label: '检测次数',
                        data: counts,
                        backgroundColor: [
                            pieChartColors[0],
                            pieChartColors[1],
                            pieChartColors[2],
                            pieChartColors[3]
                        ],
                        borderWidth: 0,
                        borderRadius: 8,
                        barPercentage: 0.6,
                        categoryPercentage: 0.7,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1000,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: '不同结果的检测个数',
                            font: {
                                size: 16,
                                weight: 'normal'
                            },
                            padding: {
                                top: 10,
                                bottom: 20
                            },
                            color: '#333'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            titleColor: '#333',
                            bodyColor: '#666',
                            borderColor: 'rgba(200, 200, 200, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            displayColors: true,
                            usePointStyle: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                drawBorder: false,
                                color: ctx => ctx.index === 0 ? 'rgba(0, 0, 0, 0)' : 'rgba(200, 200, 200, 0.1)'
                            },
                            ticks: {
                                font: {
                                    size: 12
                                },
                                color: '#555',
                                padding: 10
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                font: {
                                    size: 12
                                },
                                color: '#555',
                                padding: 10
                            }
                        }
                    }
                },
                plugins: [{
                    id: 'shadow',
                    beforeRender: (chart) => {
                        const ctx = chart.ctx;
                        ctx.save();
                        ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
                        ctx.shadowBlur = 10;
                        ctx.shadowOffsetX = 0;
                        ctx.shadowOffsetY = 4;
                        ctx.restore();
                    }
                }]
            });
        }
        
        // 更新用户检测数量图表
        if (userDetectionChartRef.current) {
            const ctx = userDetectionChartRef.current.getContext('2d');
            if (userDetectionChartInstance.current) {
                userDetectionChartInstance.current.destroy();
            }
            
            // 处理数据
            const labels = data.user_detection_count.map(item => item.username);
            const counts = data.user_detection_count.map(item => item.count);
            
            // 创建图表 - 转换为饼图
            userDetectionChartInstance.current = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: counts,
                        backgroundColor: pieChartColors,
                        borderWidth: 0,
                        borderRadius: 4,
                        hoverOffset: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '60%',
                    animation: {
                        animateRotate: true,
                        animateScale: true,
                        duration: 1000,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 15,
                                usePointStyle: true,
                                pointStyle: 'circle',
                                color: '#333'
                            }
                        },
                        title: {
                            display: true,
                            text: '不同用户的预测个数',
                            font: {
                                size: 16,
                                weight: 'normal'
                            },
                            padding: {
                                top: 10,
                                bottom: 20
                            },
                            color: '#333'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            titleColor: '#333',
                            bodyColor: '#666',
                            borderColor: 'rgba(200, 200, 200, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            callbacks: {
                                label: function(context) {
                                    let label = context.label || '';
                                    let value = context.raw || 0;
                                    let percentage = context.parsed ? 
                                        (context.parsed * 100 / context.dataset.data.reduce((a, b) => a + b, 0)).toFixed(2) + '%' : 
                                        '';
                                    return `${label}: ${value} (${percentage})`;
                                }
                            }
                        }
                    }
                },
                plugins: [{
                    id: 'doughnutLabels',
                    afterDraw: (chart) => {
                        const { ctx, data } = chart;
                        ctx.save();
                        const centerX = chart.chartArea.left + chart.chartArea.width / 2;
                        const centerY = chart.chartArea.top + chart.chartArea.height / 2;
                        
                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                        
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.font = '16px sans-serif';
                        ctx.fillStyle = '#333';
                        ctx.fillText('总检测量', centerX, centerY - 10);
                        
                        ctx.font = 'bold 24px sans-serif';
                        ctx.fillStyle = '#5e72e4';
                        ctx.fillText(total, centerX, centerY + 15);
                        
                        ctx.restore();
                    }
                }]
            });
        }
        
        // 更新用户置信度图表 - 改为雷达图
        if (userConfidenceChartRef.current) {
            const ctx = userConfidenceChartRef.current.getContext('2d');
            if (userConfidenceChartInstance.current) {
                userConfidenceChartInstance.current.destroy();
            }
            
            // 处理数据 - 只取置信度最高的六个用户
            const topUsers = data.user_confidence.slice(0, 6);
            const labels = topUsers.map(item => item.username);
            const confidences = topUsers.map(item => item.confidence);
            
            // 准备数据标签和值
            const chartData = [];
            topUsers.forEach(item => {
                chartData.push({
                    username: item.username,
                    confidence: item.confidence,
                    detection_count: item.detection_count
                });
            });
            
            // 创建图表
            userConfidenceChartInstance.current = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '置信度百分比',
                        data: confidences,
                        backgroundColor: 'rgba(94, 114, 228, 0.4)',
                        borderColor: 'rgba(94, 114, 228, 0.8)',
                        borderWidth: 2,
                        pointBackgroundColor: '#fff',
                        pointBorderColor: 'rgba(94, 114, 228, 0.8)',
                        pointHoverBackgroundColor: 'rgba(94, 114, 228, 1)',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1000,
                        easing: 'easeOutQuart'
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                backdropColor: 'rgba(255, 255, 255, 0.5)',
                                backdropPadding: 2,
                                color: '#555',
                                font: {
                                    size: 10
                                },
                                stepSize: 20,
                                showLabelBackdrop: true
                            },
                            grid: {
                                color: 'rgba(200, 200, 200, 0.2)'
                            },
                            angleLines: {
                                color: 'rgba(200, 200, 200, 0.3)',
                                lineWidth: 1
                            },
                            pointLabels: {
                                font: {
                                    size: 12,
                                    weight: 'bold'
                                },
                                color: '#333',
                                padding: 15
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: '置信度最高的六位用户',
                            font: {
                                size: 16,
                                weight: 'normal'
                            },
                            padding: {
                                top: 10,
                                bottom: 20
                            },
                            color: '#333'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            titleColor: '#333',
                            bodyColor: '#666',
                            borderColor: 'rgba(200, 200, 200, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            callbacks: {
                                label: function(context) {
                                    const item = chartData[context.dataIndex];
                                    return [
                                        `用户: ${item.username}`,
                                        `置信度: ${item.confidence.toFixed(2)}%`,
                                        `检测数量: ${item.detection_count}`
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // 更新近期检测图表
        if (recentDetectionsChartRef.current) {
            const ctx = recentDetectionsChartRef.current.getContext('2d');
            if (recentDetectionsChartInstance.current) {
                recentDetectionsChartInstance.current.destroy();
            }
            
            // 处理数据
            const labels = data.recent_detections.map(item => item.date);
            const counts = data.recent_detections.map(item => item.count);
            
            // 创建图表
            recentDetectionsChartInstance.current = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels.reverse(),
                    datasets: [{
                        label: '检测数量',
                        data: counts.reverse(),
                        borderColor: chartColors.blue.primary,
                        backgroundColor: function(context) {
                            const chart = context.chart;
                            const {ctx, chartArea} = chart;
                            if (!chartArea) {
                                return chartColors.blue.light;
                            }
                            return generateGradient(
                                ctx, 
                                chartArea, 
                                chartColors.blue.gradient[0], 
                                chartColors.blue.gradient[1]
                            );
                        },
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#fff',
                        pointBorderColor: chartColors.blue.primary,
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: chartColors.blue.primary,
                        pointHoverBorderColor: '#fff',
                        pointHoverBorderWidth: 2,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1000,
                        easing: 'easeOutQuart'
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                drawBorder: false,
                                color: ctx => ctx.index === 0 ? 'rgba(0, 0, 0, 0)' : 'rgba(200, 200, 200, 0.1)'
                            },
                            ticks: {
                                precision: 0,
                                font: {
                                    size: 12
                                },
                                color: '#555',
                                padding: 10
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                font: {
                                    size: 12
                                },
                                color: '#555',
                                padding: 10
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: '近十日预测数量',
                            font: {
                                size: 16,
                                weight: 'normal'
                            },
                            padding: {
                                top: 10,
                                bottom: 20
                            },
                            color: '#333'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            titleColor: '#333',
                            bodyColor: '#666',
                            borderColor: 'rgba(200, 200, 200, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            },
                            displayColors: true,
                            boxWidth: 8,
                            boxHeight: 8,
                            boxPadding: 4,
                            usePointStyle: true
                        }
                    }
                },
                plugins: [{
                    id: 'shadow',
                    beforeDraw: (chart) => {
                        const ctx = chart.canvas.getContext('2d');
                        ctx.save();
                        ctx.shadowColor = 'rgba(0, 0, 0, 0.1)';
                        ctx.shadowBlur = 10;
                        ctx.shadowOffsetX = 0;
                        ctx.shadowOffsetY = 4;
                        ctx.restore();
                    }
                }]
            });
        }
        
        // 更新检测来源图表
        if (detectionSourceChartRef.current) {
            const ctx = detectionSourceChartRef.current.getContext('2d');
            if (detectionSourceChartInstance.current) {
                detectionSourceChartInstance.current.destroy();
            }
            
            // 处理数据
            const labels = data.detection_by_source.map(item => item.page_type);
            const counts = data.detection_by_source.map(item => item.count);
            
            // 创建图表
            detectionSourceChartInstance.current = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: counts,
                        backgroundColor: pieChartColors,
                        borderWidth: 0,
                        borderRadius: 4,
                        hoverOffset: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    animation: {
                        animateRotate: true,
                        animateScale: true,
                        duration: 1000,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 15,
                                usePointStyle: true,
                                pointStyle: 'circle',
                                color: '#333'
                            }
                        },
                        title: {
                            display: true,
                            text: '检测来源统计',
                            font: {
                                size: 16,
                                weight: 'normal'
                            },
                            padding: {
                                top: 10,
                                bottom: 20
                            },
                            color: '#333'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            titleColor: '#333',
                            bodyColor: '#666',
                            borderColor: 'rgba(200, 200, 200, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            callbacks: {
                                label: function(context) {
                                    let label = context.label || '';
                                    let value = context.raw || 0;
                                    let percentage = context.parsed ? 
                                        (context.parsed * 100 / context.dataset.data.reduce((a, b) => a + b, 0)).toFixed(2) + '%' : 
                                        '';
                                    return `${label}: ${value} (${percentage})`;
                                }
                            }
                        }
                    }
                },
                plugins: [{
                    id: 'sourceLabels',
                    afterDraw: (chart) => {
                        const { ctx, data } = chart;
                        ctx.save();
                        const centerX = chart.chartArea.left + chart.chartArea.width / 2;
                        const centerY = chart.chartArea.top + chart.chartArea.height / 2;
                        
                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                        
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.font = '16px sans-serif';
                        ctx.fillStyle = '#333';
                        ctx.fillText('总来源数', centerX, centerY - 10);
                        
                        ctx.font = 'bold 24px sans-serif';
                        ctx.fillStyle = '#5e72e4';
                        ctx.fillText(total, centerX, centerY + 15);
                        
                        ctx.restore();
                    }
                }]
            });
        }
    }, [data]);

    // 数据更新后更新图表
    React.useEffect(() => {
        if (data) {
            updateCharts();
        }
    }, [data, updateCharts]);

    // 手动刷新数据
    const handleRefresh = () => {
        fetchData();
    };

    // 格式化置信度显示
    const formatConfidence = (value) => {
        if (value === null || value === undefined) return "未知";
        if (typeof value === 'string') {
            if (value.endsWith('%')) return value;
            try {
                value = parseFloat(value);
            } catch (e) {
                return "未知";
            }
        }
        return `${(value * 100).toFixed(2)}%`;
    };

    // 添加获取当前页记录的函数
    const getCurrentPageRecords = () => {
        if (!data || !data.realtime_detections) return [];
        
        const startIndex = currentPage * recordsPerPage;
        return data.realtime_detections.slice(startIndex, startIndex + recordsPerPage);
    };

    // 渲染加载中状态
    if (loading && !data) {
        return (
            <div className="loading-overlay">
                <div className="loading-spinner"></div>
            </div>
        );
    }

    // 渲染错误状态
    if (error && !data) {
        return (
            <div className="alert alert-danger">
                <h4>加载失败</h4>
                <p>{error}</p>
                <button className="btn btn-primary" onClick={handleRefresh}>重试</button>
            </div>
        );
    }

    return (
        <div className={`dashboard ${loading ? 'refreshing' : ''}`}>
            {loading && (
                <div className="loading-overlay">
                    <div className="loading-spinner"></div>
                </div>
            )}
            
            <div className="dashboard-header d-flex justify-content-between align-items-center mb-4">
                <h1>数据大屏</h1>
                <div>
                    <div className="update-time mb-2">
                        最后更新: {lastUpdated ? lastUpdated.toLocaleString() : '未更新'}
                    </div>
                    <button 
                        className="btn btn-primary btn-refresh" 
                        onClick={handleRefresh}
                        disabled={loading}
                    >
                        <i className="fas fa-sync-alt"></i>
                        {loading ? '刷新中...' : '刷新数据'}
                    </button>
                </div>
            </div>
            
            <div className="row">
                <div className="col-12 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <div className="d-flex justify-content-between align-items-center">
                                <span><i className="fas fa-history"></i> 实时预测信息</span>
                                {data && data.realtime_detections && data.realtime_detections.length > recordsPerPage && (
                                    <div className="carousel-indicators">
                                        {Array.from({ length: Math.ceil(data.realtime_detections.length / recordsPerPage) }).map((_, index) => (
                                            <button
                                                key={index}
                                                type="button"
                                                className={`btn btn-sm ${currentPage === index ? 'btn-primary' : 'btn-outline-secondary'}`}
                                                style={{ margin: '0 2px', width: '30px', height: '30px', borderRadius: '50%', padding: '0' }}
                                                onClick={() => setCurrentPage(index)}
                                            >
                                                {index + 1}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="card-body" style={{minHeight: '350px'}}>
                            <div className="table-responsive">
                                <table className="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>用户名</th>
                                            <th>识别权重</th>
                                            <th>检测结果</th>
                                            <th>检测来源</th>
                                            <th>时间</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {getCurrentPageRecords().map((detection, index) => (
                                            <tr key={index} className="fade-in-row" style={{animationDelay: `${index * 0.1}s`}}>
                                                <td>
                                                    <div className="data-info">
                                                        <div className="data-info-icon" style={{
                                                            background: `linear-gradient(135deg, 
                                                                hsl(${detection.username.charCodeAt(0) % 360}, 70%, 60%), 
                                                                hsl(${(detection.username.charCodeAt(0) + 40) % 360}, 70%, 50%))`
                                                        }}>
                                                            {detection.username.charAt(0).toUpperCase()}
                                                        </div>
                                                        <span className="data-info-text">{detection.username}</span>
                                                    </div>
                                                </td>
                                                <td>
                                                    <div>
                                                        <div className="progress-bar-container">
                                                            <div 
                                                                className="progress-bar" 
                                                                style={{
                                                                    width: detection.confidence_formatted && detection.confidence_formatted !== "无" ? 
                                                                        detection.confidence_formatted.replace('%', '') + '%' : '0%'
                                                                }}
                                                            ></div>
                                                        </div>
                                                        <div>{detection.confidence_formatted}</div>
                                                    </div>
                                                </td>
                                                <td>
                                                    <span className={`badge ${detection.detection_result.includes('恶性') ? 'badge-danger' : detection.detection_result.includes('良性') ? 'badge-success' : 'badge-info'}`}>
                                                        {detection.detection_result}
                                                    </span>
                                                </td>
                                                <td>
                                                    <div className="data-info">
                                                        <i className={`fas ${
                                                            detection.page_name.includes('诊断') ? 'fa-search' : 
                                                            detection.page_name.includes('分割') ? 'fa-crop-alt' : 
                                                            detection.page_name.includes('视频') ? 'fa-video' : 
                                                            'fa-brain'
                                                        }`} style={{ color: '#5e72e4', marginRight: '8px' }}></i>
                                                        {detection.page_name}
                                                    </div>
                                                </td>
                                                <td className="time-text">{detection.detection_date_formatted}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            {data && data.realtime_detections && data.realtime_detections.length > recordsPerPage && (
                                <div className="text-center mt-3">
                                    <span className="text-muted">
                                        显示 {currentPage * recordsPerPage + 1} - {Math.min((currentPage + 1) * recordsPerPage, data.realtime_detections.length)} 条，共 {data.realtime_detections.length} 条
                                    </span>
                                </div>
                            )}
                            {data && data.realtime_detections && data.realtime_detections.length === 0 && (
                                <div className="empty-state">
                                    <i className="fas fa-chart-bar"></i>
                                    <p>暂无预测数据</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-12 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <i className="fas fa-calendar-alt"></i> 近十日预测数量
                        </div>
                        <div className="card-body chart-wrapper" style={{height: '300px'}}>
                            <canvas ref={recentDetectionsChartRef}></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <i className="fas fa-chart-pie"></i> 不同结果的检测个数
                        </div>
                        <div className="card-body chart-wrapper" style={{height: '350px'}}>
                            <canvas ref={tumorChartRef}></canvas>
                        </div>
                    </div>
                </div>
                
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <i className="fas fa-users"></i> 不同用户的预测个数
                        </div>
                        <div className="card-body chart-wrapper" style={{height: '350px'}}>
                            <canvas ref={userDetectionChartRef}></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <i className="fas fa-chart-line"></i> 不同用户间的平均置信度
                        </div>
                        <div className="card-body chart-wrapper" style={{height: '400px', position: 'relative'}}>
                            <canvas ref={userConfidenceChartRef}></canvas>
                        </div>
                    </div>
                </div>
                
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <i className="fas fa-chart-bar"></i> 检测来源统计
                        </div>
                        <div className="card-body chart-wrapper" style={{height: '400px'}}>
                            <canvas ref={detectionSourceChartRef}></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// 渲染React应用
const rootNode = document.getElementById('root');
ReactDOM.createRoot(rootNode).render(<Dashboard />);
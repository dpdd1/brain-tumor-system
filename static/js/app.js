// 自定义渐变颜色生成
const generateGradient = (ctx, chartArea, colorStart, colorEnd) => {
    const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
};

// 图表颜色主题
const chartColors = {
    purple: {
        primary: 'rgba(124, 77, 255, 1)',
        light: 'rgba(124, 77, 255, 0.2)',
        gradient: ['rgba(124, 77, 255, 0.0)', 'rgba(124, 77, 255, 0.5)']
    },
    blue: {
        primary: 'rgba(66, 133, 244, 1)',
        light: 'rgba(66, 133, 244, 0.2)',
        gradient: ['rgba(66, 133, 244, 0.0)', 'rgba(66, 133, 244, 0.5)']
    },
    teal: {
        primary: 'rgba(0, 180, 180, 1)',
        light: 'rgba(0, 180, 180, 0.2)',
        gradient: ['rgba(0, 180, 180, 0.0)', 'rgba(0, 180, 180, 0.5)']
    },
    pink: {
        primary: 'rgba(255, 82, 123, 1)',
        light: 'rgba(255, 82, 123, 0.2)',
        gradient: ['rgba(255, 82, 123, 0.0)', 'rgba(255, 82, 123, 0.5)']
    },
    amber: {
        primary: 'rgba(255, 193, 7, 1)',
        light: 'rgba(255, 193, 7, 0.2)',
        gradient: ['rgba(255, 193, 7, 0.0)', 'rgba(255, 193, 7, 0.5)']
    }
};

// 饼图配色方案
const pieChartColors = [
    'rgba(124, 77, 255, 0.85)',
    'rgba(66, 133, 244, 0.85)',
    'rgba(0, 180, 180, 0.85)',
    'rgba(255, 82, 123, 0.85)',
    'rgba(255, 193, 7, 0.85)',
    'rgba(126, 211, 33, 0.85)',
    'rgba(248, 80, 50, 0.85)',
    'rgba(148, 159, 177, 0.85)',
    'rgba(77, 83, 96, 0.85)',
    'rgba(230, 126, 34, 0.85)'
];

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
                        backgroundColor: pieChartColors,
                        borderWidth: 0,
                        borderRadius: 6,
                        barPercentage: 0.7,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
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
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                drawBorder: false,
                                color: 'rgba(200, 200, 200, 0.15)'
                            },
                            ticks: {
                                font: {
                                    size: 12
                                },
                                color: '#555'
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
                                color: '#555'
                            }
                        }
                    }
                }
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
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: counts,
                        backgroundColor: pieChartColors,
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 15,
                                usePointStyle: true,
                                pointStyle: 'circle'
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
                            }
                        },
                        tooltip: {
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
                }
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
                        label: '置信度最高的六位用户',
                        data: confidences,
                        backgroundColor: 'rgba(124, 77, 255, 0.3)',
                        borderColor: 'rgba(124, 77, 255, 0.8)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(124, 77, 255, 1)',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                backdropColor: 'transparent',
                                color: '#555',
                                font: {
                                    size: 10
                                },
                                stepSize: 20
                            },
                            grid: {
                                color: 'rgba(200, 200, 200, 0.2)'
                            },
                            angleLines: {
                                color: 'rgba(200, 200, 200, 0.2)'
                            },
                            pointLabels: {
                                font: {
                                    size: 12
                                },
                                color: '#333',
                                padding: 15,
                                borderRadius: 4,
                                backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                formatter: function(value, index) {
                                    const item = chartData[index];
                                    if (item) {
                                        return [`${value}`, `${item.confidence.toFixed(2)}%`, `检测量: ${item.detection_count}`];
                                    }
                                    return value;
                                }
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
                            }
                        },
                        tooltip: {
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
                        pointBackgroundColor: chartColors.blue.primary,
                        pointBorderColor: '#fff',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                drawBorder: false,
                                color: 'rgba(200, 200, 200, 0.15)'
                            },
                            ticks: {
                                precision: 0,
                                font: {
                                    size: 12
                                },
                                color: '#555'
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
                                color: '#555'
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
                            }
                        }
                    }
                }
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
                        hoverOffset: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                padding: 15,
                                usePointStyle: true,
                                pointStyle: 'circle'
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
                            }
                        }
                    }
                }
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
                    <div className="text-muted mb-2">
                        最后更新: {lastUpdated ? lastUpdated.toLocaleString() : '未更新'}
                    </div>
                    <button 
                        className="btn btn-primary" 
                        onClick={handleRefresh}
                        disabled={loading}
                    >
                        {loading ? '刷新中...' : '刷新数据'}
                    </button>
                </div>
            </div>
            
            <div className="row">
                <div className="col-12 mb-4">
                    <div className="card">
                        <div className="card-header">
                            <div className="d-flex justify-content-between align-items-center">
                                <span>实时预测信息</span>
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
                                            <tr key={index} className="fade-in-row" style={{animation: `fadeIn 0.5s ease-in-out ${index * 0.1}s forwards`}}>
                                                <td>{detection.username}</td>
                                                <td>{detection.confidence_formatted}</td>
                                                <td>{detection.detection_result}</td>
                                                <td>{detection.page_name}</td>
                                                <td>{detection.detection_date_formatted}</td>
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
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-12 mb-4">
                    <div className="card">
                        <div className="card-header">近十日预测数量</div>
                        <div className="card-body" style={{height: '300px'}}>
                            <canvas ref={recentDetectionsChartRef}></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">不同结果的检测个数</div>
                        <div className="card-body" style={{height: '350px'}}>
                            <canvas ref={tumorChartRef}></canvas>
                        </div>
                    </div>
                </div>
                
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">不同用户的预测个数</div>
                        <div className="card-body" style={{height: '350px'}}>
                            <canvas ref={userDetectionChartRef}></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="row">
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">不同用户间的平均置信度</div>
                        <div className="card-body" style={{height: '400px', position: 'relative'}}>
                            <canvas ref={userConfidenceChartRef}></canvas>
                            {data && data.user_confidence.slice(0, 6).map((item, index) => {
                                const confidence = item.confidence ? item.confidence.toFixed(2) + '%' : '未知';
                                return (
                                    <div key={index} className="radar-label" style={{
                                        position: 'absolute',
                                        visibility: 'hidden'
                                    }}>
                                        {item.username}<br/>{confidence}<br/>{item.detection_count}
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>
                
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header">检测来源统计</div>
                        <div className="card-body" style={{height: '400px'}}>
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
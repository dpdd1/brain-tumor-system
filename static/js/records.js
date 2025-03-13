// 治疗记录页面的JavaScript功能

document.addEventListener('DOMContentLoaded', function() {
    // 搜索功能
    document.querySelector('.btn-search').addEventListener('click', function() {
        searchRecords();
    });
    
    // 回车键触发搜索
    document.getElementById('search-records').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchRecords();
        }
    });
    
    // 初始化日期选择器
    const dateInputs = document.querySelectorAll('input[type="date"]');
    dateInputs.forEach(input => {
        // 设置默认日期为今天
        if (input.id === 'detection-date') {
            const today = new Date();
            const yyyy = today.getFullYear();
            const mm = String(today.getMonth() + 1).padStart(2, '0');
            const dd = String(today.getDate()).padStart(2, '0');
            input.value = `${yyyy}-${mm}-${dd}`;
        }
    });
});

// 搜索记录函数
function searchRecords() {
    const searchTerm = document.getElementById('search-records').value.trim();
    
    // 发送搜索请求到服务器
    fetch(`/search_records?q=${encodeURIComponent(searchTerm)}`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateRecordsTable(data.records);
        } else {
            console.error('搜索失败:', data.message);
        }
    })
    .catch(error => {
        console.error('搜索请求错误:', error);
    });
}

// 更新记录表格
function updateRecordsTable(records) {
    const tbody = document.querySelector('.records-table tbody');
    tbody.innerHTML = '';
    
    if (records.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="5" class="no-records">没有找到匹配的记录</td>';
        tbody.appendChild(tr);
        return;
    }
    
    records.forEach(record => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>
                <input type="checkbox" class="record-checkbox" data-id="${record.id}">
            </td>
            <td>
                <img src="/static/${record.initial_image}" alt="脑部扫描图像" class="record-thumbnail">
            </td>
            <td>
                <img src="/static/${record.mask_image}" alt="掩膜图像" class="record-thumbnail">
            </td>
            <td>${record.detection_date}</td>
            <td>
                <button class="btn-view" onclick="viewRecord(${record.id})">查看</button>
                <button class="btn-delete" onclick="deleteRecord(${record.id})">删除</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// 查看记录详情
function viewRecord(recordId) {
    // 获取记录详情
    fetch(`/get_record/${recordId}`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showRecordDetails(data.record);
        } else {
            console.error('获取记录详情失败:', data.message);
        }
    })
    .catch(error => {
        console.error('获取记录详情请求错误:', error);
    });
}

// 显示记录详情
function showRecordDetails(record) {
    // 创建模态框
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.id = 'record-details-modal';
    modal.style.display = 'flex';
    
    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content record-details';
    
    modalContent.innerHTML = `
        <h3>治疗记录详情</h3>
        <div class="record-details-content">
            <div class="record-details-images">
                <div class="record-detail-image">
                    <h4>原始图像</h4>
                    <img src="/static/${record.initial_image}" alt="原始图像">
                </div>
                <div class="record-detail-image">
                    <h4>掩膜图像</h4>
                    <img src="/static/${record.mask_image}" alt="掩膜图像">
                </div>
            </div>
            <div class="record-details-info">
                <p><strong>检测日期:</strong> ${record.detection_date}</p>
                <p><strong>创建时间:</strong> ${record.created_at}</p>
            </div>
        </div>
        <div class="modal-buttons">
            <button class="btn-close" onclick="closeRecordDetailsModal()">关闭</button>
        </div>
    `;
    
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
}

// 关闭记录详情模态框
function closeRecordDetailsModal() {
    const modal = document.getElementById('record-details-modal');
    if (modal) {
        document.body.removeChild(modal);
    }
}

// 删除记录
function deleteRecord(recordId) {
    if (confirm('确定要删除这条记录吗？')) {
        // 发送删除请求到服务器
        fetch('/delete_record/' + recordId, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 删除成功，刷新页面
                window.location.reload();
            } else {
                alert('删除失败：' + data.message);
            }
        })
        .catch(error => {
            console.error('删除错误:', error);
            alert('删除失败，请稍后重试');
        });
    }
}

// 全选/取消全选
function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('select-all');
    const recordCheckboxes = document.querySelectorAll('.record-checkbox');
    
    recordCheckboxes.forEach(checkbox => {
        checkbox.checked = selectAllCheckbox.checked;
    });
}

// 批量删除选中的记录
function deleteSelectedRecords() {
    const selectedCheckboxes = document.querySelectorAll('.record-checkbox:checked');
    
    if (selectedCheckboxes.length === 0) {
        alert('请至少选择一条记录');
        return;
    }
    
    if (confirm(`确定要删除选中的 ${selectedCheckboxes.length} 条记录吗？`)) {
        // 收集所有选中记录的ID
        const recordIds = Array.from(selectedCheckboxes).map(checkbox => {
            return parseInt(checkbox.getAttribute('data-id'));
        });
        
        // 发送批量删除请求
        fetch('/delete_selected_records', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ record_ids: recordIds })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(data.message);
                // 删除成功，刷新页面
                window.location.reload();
            } else {
                alert('批量删除失败：' + data.message);
            }
        })
        .catch(error => {
            console.error('批量删除错误:', error);
            alert('批量删除失败，请稍后重试');
        });
    }
}
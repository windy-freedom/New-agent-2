// 格式化数字
function formatNumber(number) {
    return number.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    });
}

// 格式化百分比
function formatPercent(number) {
    return (parseFloat(number) * 100).toFixed(2) + '%';
}

// 更新市场数据
function updateMarketData() {
    fetch('/api/market-data')
        .then(response => response.json())
        .then(data => {
            const cryptoList = document.getElementById('crypto-list');
            cryptoList.innerHTML = '';
            
            for (const [crypto, info] of Object.entries(data)) {
                const card = document.createElement('div');
                card.className = 'crypto-card';
                card.innerHTML = `
                    <h3>${crypto}</h3>
                    <p>价格: ${formatNumber(info.price)} USDT</p>
                    <p>可用数量: ${formatNumber(info.quantity)}</p>
                `;
                cryptoList.appendChild(card);
            }
        });
}

// 更新学习历史记录
function updateLearningHistory(traderId, history) {
    const traderCard = document.querySelector(`.trader-card[data-trader="${traderId}"]`);
    if (!traderCard) return;

    const learningList = traderCard.querySelector('.learning-list');
    learningList.innerHTML = '';

    // 显示最近5条学习记录
    history.slice(-5).reverse().forEach(record => {
        const li = document.createElement('li');
        li.innerHTML = `
            <div>向 <span class="source">${record.source}</span> 学习</div>
            <div>改进幅度: <span class="improvement">${formatPercent(record.improvement)}</span></div>
            <div class="time">${record.time}</div>
        `;
        learningList.appendChild(li);
    });
}

// 更新AI交易者状态
function updateTraderCards(aiPerformance, selectedTrader) {
    const traderCards = document.querySelectorAll('.trader-card');
    traderCards.forEach(card => {
        const traderId = card.dataset.trader;
        const performance = aiPerformance[traderId];
        
        // 更新成功率和交易次数
        card.querySelector('.success-rate').textContent = formatPercent(performance.success_rate);
        card.querySelector('.trade-count').textContent = performance.trade_count;
        
        // 更新学习历史
        if (performance.learning_history) {
            updateLearningHistory(traderId, performance.learning_history);
        }
        
        // 更新选中状态
        if (traderId === selectedTrader) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}

// 更新账户信息
function updateAccountInfo() {
    fetch('/api/account')
        .then(response => response.json())
        .then(data => {
            document.getElementById('balance').textContent = formatNumber(data.balance);
            
            const portfolioList = document.getElementById('portfolio-list');
            portfolioList.innerHTML = '';
            
            for (const [crypto, amount] of Object.entries(data.portfolio)) {
                if (amount > 0) {
                    const li = document.createElement('li');
                    li.textContent = `${crypto}: ${formatNumber(amount)}`;
                    portfolioList.appendChild(li);
                }
            }

            // 更新AI交易状态
            const aiStatus = document.getElementById('ai-status');
            const toggleAiBtn = document.getElementById('toggle-ai');
            
            if (data.ai_trading_active) {
                aiStatus.textContent = '运行中';
                aiStatus.className = 'active';
                toggleAiBtn.textContent = '停止AI交易';
                toggleAiBtn.classList.add('active');
            } else {
                aiStatus.textContent = '已关闭';
                aiStatus.className = 'inactive';
                toggleAiBtn.textContent = '启动AI交易';
                toggleAiBtn.classList.remove('active');
            }

            // 更新AI交易者状态
            if (data.ai_performance) {
                updateTraderCards(data.ai_performance, data.selected_trader);
            }
        });
}

// 更新交易历史
function updateTradeHistory() {
    fetch('/api/trade-history')
        .then(response => response.json())
        .then(history => {
            const tbody = document.getElementById('history-table-body');
            const currentContent = tbody.innerHTML;
            const newContent = history.reverse().map(record => `
                <tr>
                    <td>${record.time}</td>
                    <td>${record.crypto}</td>
                    <td>${record.action}</td>
                    <td>${formatNumber(record.amount)}</td>
                    <td>${formatNumber(record.price)} USDT</td>
                </tr>
            `).join('');

            // 只有当内容发生变化时才更新DOM
            if (currentContent !== newContent) {
                tbody.innerHTML = newContent;
                // 添加新记录动画效果
                const newRows = tbody.querySelectorAll('tr');
                if (newRows.length > 0) {
                    newRows[0].classList.add('new-record');
                    setTimeout(() => {
                        newRows[0].classList.remove('new-record');
                    }, 1000);
                }
            }
        });
}

// 执行交易
function executeTrade(action) {
    const crypto = document.getElementById('crypto-select').value;
    const amount = parseFloat(document.getElementById('trade-amount').value);
    
    if (!amount || amount <= 0) {
        alert('请输入有效的交易数量');
        return;
    }
    
    fetch('/api/trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            crypto,
            action,
            amount
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('trade-amount').value = '';
            updateAll();
        }
    });
}

// 切换AI交易状态
function toggleAiTrading() {
    const toggleAiBtn = document.getElementById('toggle-ai');
    const newState = toggleAiBtn.textContent === '启动AI交易';
    const selectedTrader = document.querySelector('.trader-card.selected')?.dataset.trader || 'trend_follower';
    
    fetch('/api/ai-trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            active: newState,
            trader: selectedTrader
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateAll();
        }
    });
}

// 选择AI交易者
function selectTrader(traderId) {
    // 更新选中状态
    document.querySelectorAll('.trader-card').forEach(card => {
        if (card.dataset.trader === traderId) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });

    // 如果AI交易正在运行，则切换到新选择的交易者
    const aiStatus = document.getElementById('ai-status');
    if (aiStatus.classList.contains('active')) {
        fetch('/api/ai-trading', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                active: true,
                trader: traderId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateAll();
            }
        });
    }
}

// 更新所有数据
function updateAll() {
    updateMarketData();
    updateAccountInfo();
    updateTradeHistory();
}

// 定期更新数据
let updateInterval;
function startUpdates() {
    updateAll();
    
    // 清除现有的更新间隔
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // 设置新的更新间隔（1秒）
    updateInterval = setInterval(updateAll, 1000);
}

// 页面可见性变化处理
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // 页面不可见时，减少更新频率（5秒）
        if (updateInterval) {
            clearInterval(updateInterval);
        }
        updateInterval = setInterval(updateAll, 5000);
    } else {
        // 页面可见时，恢复正常更新频率（1秒）
        if (updateInterval) {
            clearInterval(updateInterval);
        }
        updateInterval = setInterval(updateAll, 1000);
    }
});

// 页面加载完成后启动
document.addEventListener('DOMContentLoaded', () => {
    startUpdates();
    
    // 添加AI交易控制按钮事件监听
    const toggleAiBtn = document.getElementById('toggle-ai');
    toggleAiBtn.addEventListener('click', toggleAiTrading);
    
    // 添加AI交易者选择按钮事件监听
    document.querySelectorAll('.select-trader').forEach(button => {
        button.addEventListener('click', (e) => {
            const traderId = e.target.closest('.trader-card').dataset.trader;
            selectTrader(traderId);
        });
    });
});
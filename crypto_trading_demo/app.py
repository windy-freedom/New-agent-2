from flask import Flask, render_template, jsonify, request
import random
from datetime import datetime
import threading
import time
import numpy as np
from collections import deque
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 模拟虚拟货币数据
crypto_data = {
    'BTC': {'price': 45000, 'quantity': 1.0, 'history': []},
    'ETH': {'price': 3000, 'quantity': 10.0, 'history': []},
    'DOGE': {'price': 0.15, 'quantity': 1000.0, 'history': []}
}

# 模拟用户账户
user_account = {
    'balance': 100000.0,  # 初始资金
    'portfolio': {
        'BTC': 0.0,
        'ETH': 0.0,
        'DOGE': 0.0
    },
    'trade_history': []
}

class LearningRecord:
    def __init__(self, source_trader, target_trader, old_weights, new_weights, time):
        self.source_trader = source_trader
        self.target_trader = target_trader
        self.old_weights = old_weights.copy()
        self.new_weights = new_weights.copy()
        self.time = time
        self.improvement = np.mean(np.abs(new_weights - old_weights))

    def to_dict(self):
        return {
            'source': self.source_trader,
            'target': self.target_trader,
            'time': self.time.strftime('%Y-%m-%d %H:%M:%S'),
            'improvement': f'{self.improvement:.4f}',
            'weight_changes': [f'{new-old:.4f}' for new, old in zip(self.new_weights, self.old_weights)]
        }

# AI交易设置
class AITrader:
    def __init__(self, name, strategy_type):
        self.name = name
        self.strategy_type = strategy_type
        self.performance = 0.0
        self.trades = []
        self.success_rate = 0.0
        self.learning_rate = 0.2  # 增加学习率，使学习更快
        self.weights = np.random.random(5)  # 策略权重
        self.trade_history = deque(maxlen=100)  # 保存最近100次交易结果
        self.learning_history = deque(maxlen=50)  # 保存最近50次学习记录
        self.last_trade_time = {}
        self.last_learn_time = time.time()  # 记录上次学习时间
        self.min_learn_interval = 5  # 减少学习间隔到5秒
        logger.info(f"创建AI交易者: {name}, 策略类型: {strategy_type}")

    def normalize_features(self, features):
        return (features - np.mean(features)) / (np.std(features) + 1e-8)

    def calculate_success_rate(self):
        if not self.trade_history:
            return 0.0
        successful_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        return successful_trades / len(self.trade_history)

    def should_learn(self, other_trader):
        """判断是否应该向其他交易者学习"""
        current_time = time.time()
        if current_time - self.last_learn_time < self.min_learn_interval:
            return False
        
        # 降低学习门槛，只要对方成功率比自己高就学习
        if other_trader.success_rate > self.success_rate:
            logger.info(f"{self.name} 准备向 {other_trader.name} 学习 (对方成功率: {other_trader.success_rate:.2%})")
            return True
        
        return False

    def learn_from_others(self, other_traders):
        """从其他AI交易者学习"""
        current_time = time.time()
        if current_time - self.last_learn_time < self.min_learn_interval:
            return None

        # 按成功率排序，优先向最成功的交易者学习
        sorted_traders = sorted(other_traders, key=lambda x: x.success_rate, reverse=True)
        for trader in sorted_traders:
            if self.should_learn(trader):
                old_weights = self.weights.copy()
                # 学习最佳交易者的权重
                self.weights += self.learning_rate * (trader.weights - self.weights)
                # 记录学习历史
                learning_record = LearningRecord(
                    trader.name,
                    self.name,
                    old_weights,
                    self.weights,
                    datetime.now()
                )
                self.learning_history.append(learning_record)
                self.last_learn_time = current_time
                logger.info(f"{self.name} 完成学习，改进幅度: {learning_record.improvement:.4f}")
                return learning_record
        return None

    def update_performance(self, profit):
        """更新交易表现"""
        self.trade_history.append({'profit': profit, 'time': datetime.now()})
        old_rate = self.success_rate
        self.success_rate = self.calculate_success_rate()
        logger.info(f"{self.name} 更新表现 - 利润: {profit:.2f}, 成功率: {self.success_rate:.2%} (之前: {old_rate:.2%})")

    def get_learning_history(self):
        """获取学习历史记录"""
        return [record.to_dict() for record in self.learning_history]

# 创建多个AI交易者
ai_traders = {
    'trend_follower': AITrader('趋势跟踪者', 'trend'),
    'mean_reversal': AITrader('均值回归者', 'reversal'),
    'volume_based': AITrader('量价分析师', 'volume'),
}

# AI交易设置
ai_trading = {
    'active': False,
    'selected_trader': 'trend_follower'  # 默认选择趋势跟踪者
}

def extract_features(crypto, history):
    """提取交易特征"""
    if len(history) < 10:
        return None
    
    prices = np.array(history[-10:])
    returns = np.diff(prices) / prices[:-1]
    
    features = np.array([
        np.mean(returns),  # 平均收益率
        np.std(returns),   # 波动率
        (prices[-1] - np.mean(prices)) / np.std(prices),  # 价格Z分数
        np.percentile(prices, 75) - np.percentile(prices, 25),  # 价格区间
        len([r for r in returns if r > 0]) / len(returns)  # 上涨比例
    ])
    
    return features

def analyze_market(crypto, trader):
    """AI交易策略分析"""
    if len(crypto_data[crypto]['history']) < 10:
        return None, 0
    
    features = extract_features(crypto, crypto_data[crypto]['history'])
    if features is None:
        return None, 0

    # 标准化特征
    normalized_features = trader.normalize_features(features)
    
    # 计算交易信号
    signal = np.dot(normalized_features, trader.weights)
    
    current_price = crypto_data[crypto]['price']
    logger.info(f"{trader.name} 分析 {crypto} - 信号: {signal:.4f}, 当前价格: {current_price:.2f}")
    
    # 根据不同策略类型调整信号
    if trader.strategy_type == 'trend':
        # 趋势跟踪策略
        if signal > 0.5:
            amount = min(user_account['balance'] / current_price * 0.1, crypto_data[crypto]['quantity'])
            logger.info(f"{trader.name} 决定买入 {crypto}: {amount:.6f}")
            return 'buy', amount
        elif signal < -0.5:
            amount = user_account['portfolio'][crypto] * 0.5
            logger.info(f"{trader.name} 决定卖出 {crypto}: {amount:.6f}")
            return 'sell', amount
    
    elif trader.strategy_type == 'reversal':
        # 均值回归策略
        if signal < -0.5:
            amount = min(user_account['balance'] / current_price * 0.1, crypto_data[crypto]['quantity'])
            logger.info(f"{trader.name} 决定买入 {crypto}: {amount:.6f}")
            return 'buy', amount
        elif signal > 0.5:
            amount = user_account['portfolio'][crypto] * 0.5
            logger.info(f"{trader.name} 决定卖出 {crypto}: {amount:.6f}")
            return 'sell', amount
    
    elif trader.strategy_type == 'volume':
        # 量价分析策略
        if abs(signal) > 0.8:
            if signal > 0:
                amount = min(user_account['balance'] / current_price * 0.15, crypto_data[crypto]['quantity'])
                logger.info(f"{trader.name} 决定买入 {crypto}: {amount:.6f}")
                return 'buy', amount
            else:
                amount = user_account['portfolio'][crypto] * 0.6
                logger.info(f"{trader.name} 决定卖出 {crypto}: {amount:.6f}")
                return 'sell', amount
    
    return None, 0

def ai_trading_loop():
    """AI交易主循环"""
    while True:
        if ai_trading['active'] and ai_trading['selected_trader'] in ai_traders:
            selected_trader = ai_traders[ai_trading['selected_trader']]
            logger.info(f"AI交易循环 - 当前交易者: {selected_trader.name}")
            
            for crypto in crypto_data:
                # 减少交易间隔时间到10秒
                if crypto not in selected_trader.last_trade_time or \
                   time.time() - selected_trader.last_trade_time.get(crypto, 0) > 10:
                    
                    action, amount = analyze_market(crypto, selected_trader)
                    if action and amount > 0:
                        initial_balance = user_account['balance']
                        initial_portfolio = user_account['portfolio'][crypto]
                        
                        # 执行交易
                        if action == 'buy':
                            total_cost = crypto_data[crypto]['price'] * amount
                            if total_cost <= user_account['balance']:
                                user_account['balance'] -= total_cost
                                user_account['portfolio'][crypto] += amount
                                selected_trader.last_trade_time[crypto] = time.time()
                                
                                # 记录交易历史
                                user_account['trade_history'].append({
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'crypto': crypto,
                                    'action': f'AI{selected_trader.name}买入',
                                    'amount': amount,
                                    'price': crypto_data[crypto]['price']
                                })
                                logger.info(f"交易执行成功 - {selected_trader.name} 买入 {crypto}: {amount:.6f} @ {crypto_data[crypto]['price']:.2f}")
                        
                        elif action == 'sell':
                            if amount <= user_account['portfolio'][crypto]:
                                total_earning = crypto_data[crypto]['price'] * amount
                                user_account['balance'] += total_earning
                                user_account['portfolio'][crypto] -= amount
                                selected_trader.last_trade_time[crypto] = time.time()
                                
                                # 记录交易历史
                                user_account['trade_history'].append({
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'crypto': crypto,
                                    'action': f'AI{selected_trader.name}卖出',
                                    'amount': amount,
                                    'price': crypto_data[crypto]['price']
                                })
                                logger.info(f"交易执行成功 - {selected_trader.name} 卖出 {crypto}: {amount:.6f} @ {crypto_data[crypto]['price']:.2f}")
                        
                        # 计算交易利润并更新AI表现
                        profit = (user_account['balance'] - initial_balance) + \
                                (user_account['portfolio'][crypto] - initial_portfolio) * crypto_data[crypto]['price']
                        selected_trader.update_performance(profit)
                        
                        # 从其他AI学习
                        other_traders = [t for name, t in ai_traders.items() if name != ai_trading['selected_trader']]
                        selected_trader.learn_from_others(other_traders)
            
        time.sleep(1)  # 减少主循环间隔到1秒

# 启动AI交易线程
ai_thread = threading.Thread(target=ai_trading_loop, daemon=True)
ai_thread.start()
logger.info("AI交易线程已启动")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market-data')
def get_market_data():
    # 模拟价格波动
    for crypto in crypto_data:
        change = random.uniform(-0.02, 0.02)  # 随机价格变动±2%
        new_price = crypto_data[crypto]['price'] * (1 + change)
        crypto_data[crypto]['price'] = new_price
        crypto_data[crypto]['history'].append(new_price)
        # 只保留最近100个价格记录
        if len(crypto_data[crypto]['history']) > 100:
            crypto_data[crypto]['history'] = crypto_data[crypto]['history'][-100:]
    
    return jsonify(crypto_data)

@app.route('/api/account')
def get_account():
    # 添加AI交易者的性能数据
    ai_performance = {name: {
        'success_rate': trader.success_rate,
        'trade_count': len(trader.trade_history),
        'learning_history': trader.get_learning_history()
    } for name, trader in ai_traders.items()}
    
    return jsonify({
        **user_account,
        'ai_trading_active': ai_trading['active'],
        'selected_trader': ai_trading['selected_trader'],
        'ai_performance': ai_performance
    })

@app.route('/api/trade', methods=['POST'])
def trade():
    data = request.json
    crypto = data['crypto']
    action = data['action']
    amount = float(data['amount'])
    
    if crypto not in crypto_data:
        return jsonify({'error': '无效的货币类型'}), 400
    
    price = crypto_data[crypto]['price']
    
    if action == 'buy':
        total_cost = price * amount
        if total_cost > user_account['balance']:
            return jsonify({'error': '余额不足'}), 400
        user_account['balance'] -= total_cost
        user_account['portfolio'][crypto] += amount
        
    elif action == 'sell':
        if amount > user_account['portfolio'][crypto]:
            return jsonify({'error': '持仓不足'}), 400
        total_earning = price * amount
        user_account['balance'] += total_earning
        user_account['portfolio'][crypto] -= amount
    
    # 记录交易历史
    user_account['trade_history'].append({
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'crypto': crypto,
        'action': '买入' if action == 'buy' else '卖出',
        'amount': amount,
        'price': price
    })
    
    return jsonify({
        'success': True,
        'account': user_account
    })

@app.route('/api/ai-trading', methods=['POST'])
def toggle_ai_trading():
    data = request.json
    if 'trader' in data and data['trader'] in ai_traders:
        ai_trading['selected_trader'] = data['trader']
        logger.info(f"切换AI交易者到: {data['trader']}")
    ai_trading['active'] = data['active']
    logger.info(f"AI交易状态: {'启动' if data['active'] else '停止'}")
    return jsonify({
        'success': True,
        'ai_trading_active': ai_trading['active'],
        'selected_trader': ai_trading['selected_trader']
    })

@app.route('/api/trade-history')
def get_trade_history():
    return jsonify(user_account['trade_history'])

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
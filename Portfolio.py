class Portfolio:
    BUY = 1
    SELL = -1
    HOLD = 0

    def __init__(self, stocks, initial_balance=10000):
        self.stocks = stocks
        self.investment = {stock: 0 for stock in stocks}
        self.investment_shares = {stock: 0 for stock in stocks}
        self.total_investment = 0
        self.investment_percentage = {stock: 0 for stock in stocks}
        self.balance = initial_balance
        self.initial_balance = initial_balance

    def reset(self):
        self.investment = {stock: 0 for stock in self.stocks}
        self.total_investment = 0
        self.investment_percentage = {stock: 0 for stock in self.stocks}
        self.balance = self.initial_balance

    def allowed_actions(self):
        return [Portfolio.BUY, Portfolio.SELL, Portfolio.HOLD]

    def take_action(self, action, stock, amount, price):
        if action == Portfolio.BUY:
            self.buy(stock, amount, price)
        elif action == Portfolio.SELL:
            self.sell(stock, amount, price)
        else:
            self.hold(stock)

    def buy(self, stock, amount, price):
        if self.balance >= amount * price:
            self.investment[stock] += amount
            self.total_investment += amount
            self.investment_shares[stock] += amount
            self.balance -= amount * price
        self.update_investment_percentage()

    def sell(self, stock, amount, price):
        if self.investment[stock] >= amount:
            self.investment[stock] -= amount
            self.total_investment -= amount
            self.investment_shares[stock] -= amount
            self.balance += amount * price
        self.update_investment_percentage()

    def update_investment_percentage(self):
        for stock in self.stocks:
            if self.total_investment > 0:
                self.investment_percentage[stock] = self.investment[stock] / self.total_investment

    def hold(self, stock):
        return self.investment[stock]

    def report(self):
        print(f"Total investment: {self.total_investment}")
        print(f"Total balance: {self.balance}")
        print(f"Total value: {self.total_investment + self.balance}")
        print(f"Investment in each stock: {self.investment}")
        print(f"Number of shares in each stock: {self.investment_shares}")


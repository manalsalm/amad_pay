import json 

class ForecastCategories(object):

    def __init__(self, category, date, predict_spending, low_spending, high_spending):
        self.category = category
        self.date = date
        self.predict_spending = predict_spending
        self.low_spending = low_spending
        self.high_spending = high_spending
        
    def __str__(self):
        '''''
        print(f"Date: {date}")
            print(f"  Predicted spending: RS{predicted:.2f}")
            print(f"  Likely range: RS{lower:.2f} to ${upper:.2f}")
            print(f"  Explanation: You are expected to spend around RS{predicted:.2f} on {category} on {date}.")
            print(f"               The spending could be as low as RS{lower:.2f} or as high as RS{upper:.2f}.")
            print()
            '''
        result = f"""Date: {self.date}:
            Predicted spending: RS{self.predict_spending:.2f}
            Likely range: RS{self.low_spending:.2f} to RS{self.high_spending:.2f}
            Explanation: You are expected to spend around RS{self.predict_spending:.2f} on {self.predict_spending} on {self.date}.
                        The spending could be as low as RS{self.low_spending:.2f} or as high as RS{self.high_spending:.2f}.
        """
        return result
    def to_dict(self):
      return {"Date": self.date, "Category": self.category, "Predict" : self.predict_spending, "Lower": self.low_spending, "Higher": self.high_spending} 
  
    def __repr__(self):
        return f'Ocean(\'{self.name}\', {self.age})'
    
    def __json__(self):
        return {
            'Date': self.date,
            'Category': self.category,
            'Predict' : self.predict_spending,
            'Lower' : self.low_spending,
            'Higher' : self.high_spending,
            '__python__': 'mymodule.submodule:ForecastCategories.from_json',
        }

    @classmethod
    def from_json(cls, json):
        obj = cls()
        obj.date = json['Date']
        obj.category = json['Category']
        obj.predict_spending = json['Predict']
        obj.low_spending = json['Lower']
        obj.high_spending = json['Higher']
        return obj
from sklearn.ensemble import GradientBoostingClassifier

def test_user_data(data, X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict([data])
    return y_pred
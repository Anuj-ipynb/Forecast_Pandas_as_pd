import matplotlib.pyplot as plt

def plot_feature_importance(model, features):
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title("Feature Importance")
    return fig
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf= LogisticRegression(solver="liblinear",random_state=42)
rnd_clf =RandomForestClassifier(n_estimators=10,random_state=42)
svm_clf =SVC(gamma="auto",random_state=42)

from sklearn.ensemble import VotingClassifier
voting_clf =VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    voting='hard')

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

from sklearn.datasets import make_moons
X,y =make_moons(n_samples=500,noise=0.30,random_state=42)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Bagging
bag_clf =BaggingClassifier(
    DecisionTreeClassifier(random_state=42),n_estimators=500,
    max_samples=100,bootstrap=True,n_jobs=-1,random_state=42)
bag_clf.fit(X_train,y_train)
y_pred =bag_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

tree_clf =DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train,y_train)
y_pred_tree= tree_clf.predict(X_test)
print(accuracy_score(y_test,y_pred_tree))

#plotDB
#for plot decision boundary
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf,X,y)
plt.title("Decision Tree",fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf,X,y)
plt.title("Decision Trees with Bagging",fontsize=14)
#Exp3_plot.save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()



from sklearn.ensemble import AdaBoostClassifier
#AdaBoost
ada_clf= AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),n_estimators=200,
    algorithm="SAMME.R",learning_rate=0.5,random_state=42)
    #基于2g0个单层决策树max depth:=I,就是一个决策节点加两个叶节点。
    #这是AdaBoostClassifier默认使用的基础估算器
ada_clf.fit(X_train,y_train)
y_pred= ada_clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
plot_decision_boundary(ada_clf,X,y)
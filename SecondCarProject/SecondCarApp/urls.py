from django.urls import path

from . import views

app_name = 'SecondCarApp'
urlpatterns = [
    path('home', views.home, name='home'),
    path('available-cars', views.available_cars, name='available-cars'),
    path('show-more/<int:id>', views.show_more, name='show-more'),
    path('predict-value/<int:id>',views.predict_value, name='predict-value'),
    path('gallery',views.gallery, name='gallery'),
    path('accuracy',views.user_accuracy, name='accuracy'),
    path('analysis',views.analysis, name='analysis'),
    path('bar-graph-1',views.bar_graph_1, name='bar-graph-1'),
    path('bar-graph-2',views.bar_graph_2, name='bar-graph-2'),
    path('pie-graph-1',views.pie_graph_1, name='pie-graph-1'),
    path('model-details',views.model_details, name='model-details'),
    path('back-to-model-btn',views.back_to_model_btn, name='back-to-model-btn'),
    path('back-to-accuracy-btn',views.back_to_accuracy_btn, name='back-to-accuracy-btn')

]
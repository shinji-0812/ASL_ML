from django.urls import path
from . import views
from . import predict as prediction_views

urlpatterns = [
    path("", views.index, name="index"),
    path("predict/", prediction_views.predict, name="predict"),
]
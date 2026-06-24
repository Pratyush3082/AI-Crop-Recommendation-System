from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),              # site root shows login page
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    path("predict/", views.predict_view, name="predict"),  # protected view; user goes here after login
    path("api/predict/", views.predict_api, name="predict_api"),
]
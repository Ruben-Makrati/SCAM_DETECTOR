from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('results/', views.results_analysis, name='results_analysis'),
    path('history/', views.detection_history, name='detection_history'),
    path('report/', views.report_scam, name='report_scam'),
    path('performance/', views.model_performance, name='model_performance'),
    path('about/', views.about, name='about'),
    path('how-it-works/', views.how_it_works, name='how_it_works'),
    path('profile/', views.profile, name='profile'),
    path('login/', LoginView.as_view(template_name='detector/login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('register/', views.register, name='register'),
    
    # API endpoints
    path('api/detect/', views.api_detect, name='api_detect'),
    path('api/statistics/', views.api_statistics, name='api_statistics'),
    path('api/mark_feedback/', views.mark_detection_feedback, name='mark_detection_feedback'),
    path('clear_history/', views.clear_history, name='clear_history'),
] 
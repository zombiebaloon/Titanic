from django.contrib import admin
from django.urls import path

from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('result/', result, name='result'),
    path('EDA/', EDA,name='EDA'),
]


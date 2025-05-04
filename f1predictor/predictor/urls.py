from django.urls import path
from .views import PredictionView, DriversListView

urlpatterns = [
    path('predict/', PredictionView.as_view(), name='predict'),
    path('drivers/<int:year>/', DriversListView.as_view(), name='drivers'),
    path('drivers/<int:year>/<int:round>/', DriversListView.as_view(), name='drivers_with_round'),
]

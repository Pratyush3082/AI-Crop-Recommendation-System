from django.db import models

class Query(models.Model):
    N = models.FloatField()
    P = models.FloatField()
    K = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    season = models.CharField(max_length=50, blank=True, null=True)
    location = models.CharField(max_length=150, blank=True, null=True)
    predicted_crop = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

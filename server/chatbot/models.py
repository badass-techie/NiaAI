from django.db import models

# Create your models here.
class User(models.Model):
    address = models.CharField(max_length=255, blank=False)
    purpose_statement = models.CharField(max_length=4095, blank=False)

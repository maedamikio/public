from django.db import models


class Log(models.Model):

    create = models.DateTimeField(auto_now_add=True)
    update = models.DateTimeField(auto_now=True)

    filename = models.CharField(max_length=100, null=True, blank=True)
    filesize = models.IntegerField(null=True, blank=True)
    s3_key = models.CharField(max_length=100, null=True, blank=True)

    message = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=100, null=True, blank=True)

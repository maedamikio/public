from django.contrib import admin

from app.models import Log


class LogAdmin(admin.ModelAdmin):

    list_display = ('id', 'create', 'status', 'filename', 'filesize', 's3_key', 'message')
    list_filter = ['create', 'status']
    search_fields = ['message']


admin.site.register(Log, LogAdmin)

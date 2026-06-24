from django.contrib import admin
from .models import Query

@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ('predicted_crop','confidence','created_at')
    readonly_fields = ('created_at',)

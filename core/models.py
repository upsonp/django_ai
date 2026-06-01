import uuid

from django.db import models


# Create your models here.
class TodoList(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()


class TodoItem(models.Model):
    list = models.ForeignKey(TodoList, on_delete=models.CASCADE, related_name='todo_items')
    priority = models.IntegerField(default=0)
    description = models.CharField(max_length=100)

    def __str__(self):
        return self.description

    class Meta:
        ordering = ['priority']

class ChatThread(models.Model):
    thread_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
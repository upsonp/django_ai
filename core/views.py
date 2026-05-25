from crispy_forms.bootstrap import StrictButton
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Row, Column, Field, Submit
from crispy_forms.utils import render_crispy_form
from django import forms
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy, path

from core.agent import todo_request
from core.agent.todo_request import make_request
from core.models import TodoList, TodoItem


class TodoListForm(forms.ModelForm):
    class Meta:
        model = TodoList
        fields = ['title', 'description']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        url = reverse_lazy('core:submit_todo')
        submit_attrs = {
            'hx-target': "#form_id_todo_form",
            'hx-swap': 'outerHTML',
            'hx-post': url,
        }

        submit = StrictButton('<span class="bi bi-plus-square"></span> Save', type='button', css_class='btn btn-primary', **submit_attrs)
        self.helper = FormHelper()
        self.helper.form_id = 'form_id_todo_form'
        self.helper.layout = Layout(
            Row(
                Column(Field('title'))
            ),
            Row(
                Column(Field('description'))
            ),
            Row(
                Column(submit)
            ),
        )

# Create your views here.
def index(request):
    form = TodoListForm()
    todo_lists = TodoList.objects.all()
    template = render(request, 'core/index.html', context={'todo_form': form, 'todo_lists': todo_lists})
    return HttpResponse(template)


def submit_todo(request):
    print("Submitted TodoList")
    triggers = []
    form = TodoListForm(request.POST)
    if form.is_valid():
        print("Valid")
        todo_list = form.save()

        form = TodoListForm()
        triggers.append("update_todo_lists")
    else:
        print(f"Invalid form: {form.errors}")

    html = render_crispy_form(form)
    response = HttpResponse(html)
    response['HX-Triggers'] = ','.join(triggers)
    return response


def list_todo_lists(request):
    html = render(request, 'core/partials/todo_lists.html', {'todo_lists': TodoList.objects.all()})
    return HttpResponse(html)


def update_todo_lists(request):
    pass


def generate_todo_items(request, todo_list_id):
    todo_list = TodoList.objects.get(id=todo_list_id)

    new_todo_items = make_request(todo_list.title, todo_list.description)
    items = [
        TodoItem(list_id=todo_list_id, description=item) for item in new_todo_items
    ]
    TodoItem.objects.bulk_create(items)

    html = render(request, 'core/partials/todo_lists.html#new-items', {'todo_list': todo_list, 'todo_items': items})
    response = HttpResponse(html)
    return response


urlpatterns = [
    path('', index, name='index'),
    path('todo/list/', list_todo_lists, name='list_todo_lists'),
    path('submit_todo/', submit_todo, name='submit_todo'),
    path('todo/generate/<int:todo_list_id>', generate_todo_items, name='generate_todo_items'),
]
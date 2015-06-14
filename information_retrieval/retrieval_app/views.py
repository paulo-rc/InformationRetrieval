from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

def index(request):
  return HttpResponse(dictionary)
  return render(request, 'retrieval_app/index.html')

import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage

def tu_vista_de_procesamiento(request):
    if request.method == 'POST':
        archivo_inmail = request.FILES.get('inmail_file')
        archivo_arff = request.FILES.get('arff_file')

        if archivo_inmail and archivo_arff:
            # 1. Guardar archivos temporalmente en /tmp/ (Railway permite esto)
            path_inmail = os.path.join('/tmp', archivo_inmail.name)
            path_arff = os.path.join('/tmp', archivo_arff.name)

            with open(path_inmail, 'wb+') as destination:
                for chunk in archivo_inmail.chunks():
                    destination.write(chunk)

            with open(path_arff, 'wb+') as destination:
                for chunk in archivo_arff.chunks():
                    destination.write(chunk)

            # 2. LLAMADA A TUS MÓDULOS
            try:
                # Ejemplo: Módulo 05 procesa inmail
                # resultado_m5 = modulo05.procesar(path_inmail)
                
                # Ejemplo: Módulos 06-09 procesan arff
                # resultado_m6_09 = modulo06.procesar(path_arff)
                
                return render(request, 'dashboard/success.html', {
                    'mensaje': 'Archivos procesados correctamente'
                })
            except Exception as e:
                return render(request, 'dashboard/error.html', {'error': str(e)})
            finally:
                # 3. Limpieza: Borrar archivos temporales para no llenar el disco
                if os.path.exists(path_inmail): os.remove(path_inmail)
                if os.path.exists(path_arff): os.remove(path_arff)

    return render(request, 'dashboard/upload.html')


# Piloto D08 — ULB Uniform con m=1000

## Objetivo

Comprobar si TabPFN-2.5 puede utilizar un contexto de 1000
transacciones y medir el tiempo y la memoria necesarios.

## Problema del desbalance

ULB contiene 284807 transacciones, pero solo 492 son fraudes.

En la partición de entrenamiento de la semilla 0 tenemos:
- 170884 transacciones.
- 295 fraudes.
- Una proporción de fraude de aproximadamente 0,17 %.

Con Uniform y m=1000 esperamos encontrar, en promedio,
1,73 transacciones fraudulentas.

La probabilidad teórica de obtener un contexto sin ningún
fraude es aproximadamente 17,68 %.

Por eso, algunos contextos pueden contener cero positivos.

## Qué hacemos cuando ocurre

Si el contexto contiene una sola clase, no ejecutamos TabPFN.

El script registra el caso en pilot_log.csv con el estado
"skipped_one_class".

Esto evita intentar clasificar utilizando un contexto donde
no existe ningún ejemplo de fraude.

## Comparación con Balanced

Balanced intenta seleccionar 500 fraudes y 500 no fraudes.

Pero en el entrenamiento de ULB solo hay 295 fraudes.

Por lo tanto, selecciona:
- 295 fraudes.
- 705 transacciones normales.

El contexto queda con 29,5 % de positivos.

Se registra capped=True porque no había suficientes
fraudes para alcanzar el balance 50/50.

## Resultado del piloto

Las mediciones reales se guardan en results/pilot_log.csv.

Se registran el tiempo de fit, el tiempo de predicción,
los segundos por cada 1000 filas y la memoria máxima
utilizada por PyTorch.

Estos resultados permitirán decidir qué configuración
conviene utilizar en los experimentos posteriores.

---
status: accepted (provisional)
---

# Texturizado por proyección multivista sobre ComfyUI: seguimos, con condiciones

Veredicto del spike #3 (tickets #17, #18 y #19). El pipeline sigue así: Blender renderiza depth/normal, ComfyUI (SDXL + ControlNet union + IP-Adapter) pinta cada vista continuando la pintura de las anteriores (repaint con máscara suave y differential diffusion), se proyecta y se hornea a la UV con numpy, se mezcla y se reduce a PSX. **Decisión: seguimos con este enfoque para la textura de cuerpo (#11), con las condiciones de abajo.** No se evalúa ahora un texturizador multivista de malla existente (Hunyuan3D-Paint o MV-Adapter). Solo se evaluará si salta alguno de los disparadores listados más abajo.

El motivo es que la parte mecánica está resuelta y es barata: no quedan huecos, se cumple el contrato PSX, el resultado es reproducible y el tiempo es de unos 2 minutos. Lo que falla es la coherencia de contenido entre vistas, y eso pesa poco en una textura de cuerpo (piel y ropa interior lisas). Pesará más en prendas (#13). Cambiar ahora de stack costaría otro spike sin un problema demostrado que lo justifique.

## Evidencia

Resultado para validar: `C:\Develop\AiCharacterCreator-work\texturing\mv19_a\test_mesh_textured.blend`. Previews: `preview_ring.png` (8 azimuts cada 45° más 2 desde arriba), `seams_closeup.png` y `generated.png` (las 5 vistas generadas y la textura), en la misma carpeta. `mv19_b` es una segunda ejecución idéntica.

Malla de prueba: un busto de 122 tris (torso de 10 lados, cabeza icosfera y nariz). Referencia: un caballero cartoon generado con SDXL (`refs\knight_bust.png`). Semilla 7, 5 vistas (front, right, left, back, top), renders de condicionamiento de 1024², bake de 512 y textura final de 128 px.

- **Calidad: buena en ropa y armadura, sin probar en caras.** A 128 px y 256 colores el torso se lee como un tabardo rojo con ribete dorado y un escudo heráldico que sigue la referencia. Tiene un look PSX creíble. La cabeza sale como una bola facetada dorada y acero, sin cara. En parte es culpa de la malla: una icosfera no tiene cuencas, nariz ni barbilla que ControlNet pueda seguir.
- **Coherencia entre vistas: media.** La paleta, los materiales (tela roja, oro, acero) y el estilo de dibujo son consistentes alrededor de toda la malla. El contenido no lo es: cada vista se inventa su emblema centrado. Delante sale un escudo, en los laterales otros escudos y detrás un adorno vertical, así que el torso acaba con 4 emblemas. Es el problema "Janus" de generar vista a vista. Usar IP-Adapter en modo "style transfer" para las vistas posteriores y añadir prompts por vista lo reduce, pero no lo elimina.
- **Costuras: casi invisibles a resolución PSX, con una excepción.** En las vistas intermedias (front_right, back_left, front_left) los relevos entre cámaras coinciden con aristas del torso y se leen como cambios de panel, no como cortes. Hay una costura visible: en back_right, un corte diagonal crema/naranja cruza una cara del torso, donde dos vistas con poca confianza no coinciden en el sombreado. La cabeza es un mosaico de facetas. A 128 px pasa por "casco facetado", pero no es una superficie limpia.
- **Iluminación horneada.** Cada vista generada trae sus propias luces y sombras (por ejemplo, las bandas grises oscuras a los lados del frente). Es la causa principal de las costuras que quedan, y en piel se notará más que en tela.
- **Cobertura, contrato y determinismo: resueltos.** El 76,3 % de la superficie lo ve alguna cámara. El resto son las tapas inferior y superior del torso, que no se ven, y la parte de abajo de la nariz (~0,5 %). Después del relleno no queda ningún texel de superficie sin pintar. La textura cumple el contrato: 128×128, como máximo 256 colores, filtrado Closest y empaquetada. Con `free()` antes de cada ejecución, dos ejecuciones dan vistas y textura idénticas bit a bit (sha `82c57e44...07088`).

### Tiempos (RTX 4080 SUPER, juego cerrado)

| paso | mv19_a (s) | mv19_b (s) |
|---|---|---|
| generar front (txt2img, en frío tras `free()`) | 30,2 | 27,5 |
| generar right (repaint, 2 IP-Adapters, cambio de grafo) | 24,3 | 24,5 |
| generar left / back / top | 18,4 / 16,7 / 15,9 | 18,5 / 17,9 / 18,0 |
| render de condicionamiento (cada vista, Cycles CPU) | 2,1–2,5 | 2,2–2,4 |
| proyección y bake (cada vista) | 0,14–0,28 | 0,13–0,28 |
| merge, relleno y reducción PSX | 2,6 | 2,7 |
| empaquetado y 10 previews | 2,1 | 2,2 |
| **total** | **125,7** | **126,8** |

La generación se lleva ~105 s (83 %): unos 16–18 s por vista en caliente y 25–30 s para las dos primeras. El resto suma ~21 s, de los que ~12 s son 6 arranques de Blender. En #17 se midió que, con Street Fighter 6 abierto (7,5 GB de VRAM ocupados), cada imagen tarda de 2 a 3 veces más por el offloading dinámico de ComfyUI. Con el juego abierto, cabe esperar unos 4–6 minutos por textura.

### VRAM

El pico es de **15,1 GB de los 16 GB** de la tarjeta, con una base de ~1,0–1,3 GB (`vram19_final.log`, muestreo cada 500 ms). Una sola vista (#17) se quedaba en 9,1 GB. El salto viene de los dos pases de IP-Adapter y del VAE encode, y además ComfyUI no devuelve lo que reserva. En la práctica, este PC no puede texturizar y jugar a la vez sin una ralentización fuerte, y ComfyUI retiene ~14 GB hasta que se libera o se para.

## Límites de la evidencia

- **No se han evaluado caras.** La cabeza es una icosfera. La cara va en un material y atlas propios generados por inpainting (#12), así que esta decisión no la cubre. Lo que sí cubre la textura de cuerpo (nuca, orejas, cuello) solo se ha visto sobre geometría muy pobre.
- **Una sola malla, una sola referencia, una sola semilla final.** El busto de 122 tris tiene poca oclusión. Un humanoide de ~1000 tris en A-pose tiene axilas, entrepierna, cara interior de los brazos y espacio entre las piernas, que 5 vistas no van a cubrir igual de bien.
- **La resolución final fue de 128 px.** #11 pide 256 px, que muestra el doble de detalle, y también el doble de costuras y de sombreado horneado.
- **La duplicación de emblemas está confirmada**, no es un caso aislado. En un cuerpo con piel y ropa interior lisas no debería notarse. En prendas con un motivo único (un logo, un escudo) sí aparecerá.
- La validación visual es mía, hecha sobre los previews renderizados. El usuario abrió el `.blend` el 2026-10-25 y aceptó el veredicto como **provisional**: la malla de prueba es demasiado abstracta para juzgar la calidad de un personaje, así que solo valida la mecánica (proyección, cobertura, costuras, contrato PSX y reproducibilidad). La calidad se valida de verdad en #11, sobre el cuerpo base real (condición 5).

## Condiciones para #11 (texturizador de cuerpo)

1. **Iluminación plana.** Hay que añadir al prompt y al negativo términos de albedo e iluminación plana ("flat shading, albedo, no shadows" / "shadows, highlights, rim light"). Si no basta, hay que probar un pase de des-iluminación (igualar la luminancia por vista antes del merge). En piel, el sombreado horneado es el riesgo número uno.
2. **Más vistas para un humanoide.** Hay que partir de 8 azimuts cada 45° más vistas en picado y contrapicado, y bajar `KEEP_FULLY` para que el repaint tenga más solape. Además, hay que medir la cobertura vista (`coverage.json`) sobre el cuerpo base real (#4/#5): la superficie visible que ninguna cámara vea debe quedar por debajo del ~2 %.
3. **VRAM.** Al acabar cada build, hay que llamar a `free(unload_models=True)` (o parar ComfyUI). `make build` debe avisar si detecta poca VRAM libre (un juego abierto) en vez de tardar en silencio. El pico de cada build se registra en `timings.json` para seguir la tendencia.
4. **Presupuesto de tiempo.** Hay que mantener la generación por debajo de ~4 minutos por textura con la GPU libre. Si 8 o más vistas lo superan, hay que reducir pasos o resolución de condicionamiento antes de recortar vistas.
5. **Validación del usuario sobre el cuerpo real.** Las condiciones anteriores se dan por cumplidas cuando el usuario abre el `.blend` de #11 con un personaje de prueba hecho a partir de una referencia real y lo acepta.

## Cuándo evaluar Hunyuan3D-Paint / MV-Adapter

Hay que abrir un ticket de evaluación (un spike corto, con la misma malla y referencia y comparando contra este pipeline) en cuanto ocurra cualquiera de estas cosas:

- en #11, la iluminación horneada o las costuras de la piel siguen siendo visibles a 256 px después de aplicar la condición 1;
- en #13 (prendas), la duplicación de motivos por vista hace inaceptable una prenda con un motivo único;
- en #12, el inpainting de la cara no da una identidad estable y se plantea pintar la cabeza por proyección.

Hay que tener en cuenta que estos modelos generan todas las vistas a la vez, lo que resolvería la coherencia de contenido. Probablemente consumen tanta VRAM como el pipeline actual o más. Antes de adoptarlos, hay que medir que quepan en 16 GB.

## Opciones consideradas

- **Generar cada vista de forma independiente y mezclar después.** Se descartó en #19: los emblemas se duplicaban y había cortes con fundido cruzado.
- **Pasar la vista frontal a las demás como referencia IP-Adapter en modo "linear".** Se descartó porque copia la composición y pone un escudo en cada vista.
- **Hunyuan3D-Paint / MV-Adapter ahora.** Se aplaza: este pipeline cumple, y los disparadores de arriba deciden cuándo merece la pena el cambio.

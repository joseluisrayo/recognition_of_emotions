$(document).ready(function () {

    $(".btn-type-camera").on('click', function () {
        var v_txt_btn = $(this).text();
        const trimmedText = v_txt_btn.replace(/\s+/g, "");
        if (trimmedText === "ActivarCámara") {
            $(this).text("Desactivar Cámara");
            $("#camara_reconcer_emotion").attr("src", "/video_stream_live");
            $("#icono-camara-not-record").hide();
            $("#btn-camara-grabar-stop").show();

        } else if (trimmedText === "DesactivarCámara") {
            $(this).text("ActivarCámara");
            $("#camara_reconcer_emotion").attr("src", "");
            $("#icono-camara-not-record").show();
            $("#btn-camara-grabar-stop").hide();
            $(".text-respuest-record").hide();

        } else if (trimmedText === "Grabar") {
            $("#ico-statu-camara").removeClass("fa-circle").addClass("fa-stop");
            $("#txt-statu-camara").text("Detener");

            $(".text-respuest-record").show();

        } else if (trimmedText === "Detener") {
            $("#ico-statu-camara").removeClass("fa-stop").addClass("fa-circle");
            $("#txt-statu-camara").text("Grabar");

            $(".text-respuest-record").hide();
        }
    });

});
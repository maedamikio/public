// https://docs.djangoproject.com/en/3.0/ref/csrf/
function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    var cookies = document.cookie.split(';');
    for (var i = 0; i < cookies.length; i++) {
      var cookie = jQuery.trim(cookies[i]);
      // Does this cookie string begin with the name we want?
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});


// クリックしてファイルを選択
$('#img').on('click', function() {
  $('#file').click();
});

$('#file').change(function() {
  var files = this.files;
  if (checkImg(files)) {
    file = files[0];
    readImg(file);
    predict(file);
  }
});


// ファイルをドラッグ&ドロップ
var img;

img = document.getElementById('img');
img.addEventListener('dragenter', dragenter, false);
img.addEventListener('dragover', dragover, false);
img.addEventListener('drop', drop, false);

function dragenter(e) {
  e.stopPropagation();
  e.preventDefault();
}

function dragover(e) {
  e.stopPropagation();
  e.preventDefault();
}

function drop(e) {
  e.stopPropagation();
  e.preventDefault();

  var dt = e.dataTransfer;
  var files = dt.files;
  if (checkImg(files)) {
    file = files[0];
    readImg(file);
    predict(file);
  }
}


// 1ファイル以上、jpeg、png、10MB以上の場合は処理をしない
function checkImg(files) {
  if (files.length != 1 ) {
    return false;
  }
  var file = files[0];
  console.log(file.name, file.size, file.type);
  if (file.type != 'image/jpeg' && file.type != 'image/png') {
    return false;
  }
  if (file.size > 10000000) {
    return false;
  }
  return true;
}


// ファイルの読み込み
function readImg(file) {
  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = function() {
    $('#img').attr('src', reader.result);
  }
}


// 推論API
function predict(file) {

  $('#img').css('display', 'none');
  $('.spinner').css('display', '');

  var formData = new FormData();

  formData.append('file', file);

  $.ajax({
    type: 'POST',
    url: '/api/',
    data: formData,
    processData: false,
    contentType: false,
    success: function(response) {
      console.log(response);
      $('#img').attr('src', response);
      $('.spinner').css('display', 'none');
      $('#img').css('display', '');
    },
    error: function(response) {
      console.log(response);
      $('#img').attr('src', '/static/abe_or_ishihara.png');
      $('.spinner').css('display', 'none');
      $('#img').css('display', '');
    }
  });
}

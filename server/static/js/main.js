function get_text(res) {
    result = Object.keys(res).map(function(key) {
      return {key: key, val: res[key]};
    });

    result = result.sort(function (a, b) {
        return b.val - a.val;
    });
    text = ''
    for (i=0; i<result.length; i++) {
        text = text + gen_list_item(result[i].key, result[i].val) + "\n" 
    }

    text = '<ul class="list-group" id="prob">' + text + '</ul>'

    return text
}

function clickAction(key) {
    input = $("#input");
    input.val(key+' ');
    input.focus();
}

function gen_list_item(key, val) {
    return '<li class="list-group-item" onclick="clickAction(\''+key+'\')">' + key + '<span class="badge">' + Number(val).toFixed(6) + '</span></li>'
}

function check_empty() {
    s = $('#input').val();
    if (s.length < 1) {
        prob = $("#prob");
        if (prob) {
            prob.remove();
        }
        return 1;
    }
    return 0;
}

$(document).ready(function(){
  $("#input").keyup(function(){
    if (check_empty()) {
        return;
    }
    s = $('#input').val();
    len = s.length;
    if (s[len-1]==' ' || s[len-1]=='^' || s[len-1]=='$') {
        $("#next").html("");
        return;
    }

    $.ajax({
        type: 'POST', 
        url: "http://localhost:5000/predict", 
        crossDomain: true,
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        data: JSON.stringify({input: s}), 
        success: function(result){
            text = get_text(result)
            $("#next").html(text);
        }
    });
  });
});
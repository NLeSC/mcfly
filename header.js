!function() {
    var path = document.location.pathname;
    var dir = /^.*\/([a-z]+)\//.exec(path)[1];
    var filename = path.substring(path.lastIndexOf('/')+1);
    document.write([
        '<div id="header">',
        '<hr>',
        '</div>'
    ].join(''));
}();

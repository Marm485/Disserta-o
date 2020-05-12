function get_rows(label){
    var results = document.getElementById("customers");
    r = 0;
    var prev = null;
    while(row = results.rows[r++]){
        var c = 0;
        while(cell = row.cells[c++]){
            if(prev == true){
                cell.style.backgroundColor = "#6699ff";
                prev = false;
            }
            else if(cell.innerHTML == label){
                cell.style.backgroundColor = "#6699ff";
                prev = true;
            }
        }
    }
}
function get_rows(id,label){
    var results = document.getElementById(id);
    r = 0;
    while(row = results.rows[r++]){
        var c = 0;
        while(cell = row.cells[c++]){
	    console.log("célula: " + cell.innerHTML);
	    console.log("label: " + label);
            if(cell.innerHTML.substring(0,(cell.innerHTML.lastIndexOf(" ")-2)) == label){
                cell.style.backgroundColor = "#6699ff";
            }
        }
    }
}

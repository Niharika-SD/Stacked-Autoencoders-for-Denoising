function nimg = normalz(img)
maxe = max(max(img)');
mine = min(min(img)');
nimg = (img - mine)/(maxe - mine + eps); 

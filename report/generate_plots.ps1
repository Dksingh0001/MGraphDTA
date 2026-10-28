param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$out = Join-Path $PSScriptRoot 'figures'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$navy = [System.Drawing.Color]::FromArgb(22,50,79)
$blue = [System.Drawing.Color]::FromArgb(40,120,165)
$teal = [System.Drawing.Color]::FromArgb(42,157,143)
$gold = [System.Drawing.Color]::FromArgb(226,151,40)
$red = [System.Drawing.Color]::FromArgb(191,75,74)
$purple = [System.Drawing.Color]::FromArgb(123,93,166)
$palette = @($navy,$blue,$teal,$gold,$red,$purple)
$font = New-Object System.Drawing.Font('Arial', 10)
$small = New-Object System.Drawing.Font('Arial', 8)
$titleFont = New-Object System.Drawing.Font('Arial', 12, [System.Drawing.FontStyle]::Bold)
$axisPen = New-Object System.Drawing.Pen($navy, 1.4)
$gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220,226,231), 1)

function New-Canvas([int]$W,[int]$H,[string]$Title) {
    $bmp = New-Object System.Drawing.Bitmap($W,$H)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::White)
    $g.DrawString($Title,$script:titleFont,[System.Drawing.Brushes]::Black,24,16)
    return @{ Bitmap=$bmp; Graphics=$g }
}
function Save-Canvas($Canvas,[string]$Name) {
    $Canvas.Bitmap.Save((Join-Path $script:out $Name),[System.Drawing.Imaging.ImageFormat]::Png)
    $Canvas.Graphics.Dispose(); $Canvas.Bitmap.Dispose()
}
function Draw-Panel($G,[int]$X,[int]$Y,[int]$W,[int]$H,[string]$Title,[string[]]$Labels,[double[]]$Values,[double]$Max,[double[]]$Errors=@(),[string]$Unit='') {
    $G.DrawString($Title,$script:titleFont,[System.Drawing.Brushes]::Black,$X,$Y)
    $l=$X+48; $r=$X+$W-10; $t=$Y+28; $b=$Y+$H-55; $ph=$b-$t; $pw=$r-$l
    for($k=0;$k -le 4;$k++) {
        $v=$Max*$k/4.0; $yy=$b-[int]($ph*$k/4.0)
        $G.DrawLine($script:gridPen,$l,$yy,$r,$yy)
        $G.DrawString(('{0:0.##}' -f $v),$script:small,[System.Drawing.Brushes]::DimGray,$X,$yy-8)
    }
    $G.DrawLine($script:axisPen,$l,$t,$l,$b); $G.DrawLine($script:axisPen,$l,$b,$r,$b)
    $n=$Values.Count; $slot=$pw/[Math]::Max(1,$n); $bw=[Math]::Min(42,$slot*0.55)
    for($i=0;$i -lt $n;$i++) {
        $ratio=[double]$Values[$i]/[double]$Max; if($ratio -lt 0){$ratio=0}; $bh=[int]($ratio*$ph); $bx=[int]($l+$slot*$i+($slot-$bw)/2); $by=$b-$bh
        $brush=New-Object System.Drawing.SolidBrush($script:palette[$i % $script:palette.Count])
        $G.FillRectangle($brush,$bx,$by,[int]$bw,$bh); $brush.Dispose()
        if($Errors.Count -eq $n) {
            $e=[int]($Errors[$i]/$Max*$ph); $cx=$bx+[int]($bw/2)
            $G.DrawLine($script:axisPen,$cx,$by-$e,$cx,$by+$e); $G.DrawLine($script:axisPen,$cx-5,$by-$e,$cx+5,$by-$e); $G.DrawLine($script:axisPen,$cx-5,$by+$e,$cx+5,$by+$e)
        }
        $lab=$Labels[$i]; $sz=$G.MeasureString($lab,$script:small); $G.DrawString($lab,$script:small,[System.Drawing.Brushes]::Black,[single]($l+$slot*$i+($slot-$sz.Width)/2),[single]($b+6))
    }
    if($Unit){$G.DrawString($Unit,$script:small,[System.Drawing.Brushes]::DimGray,$r-55,$t-16)}
}

# Training/validation trajectories, directly read from the saved epoch CSVs.
$series = @(
    @{Name='Primary raw, batch 32 (manual stop)';Path='results/davis/run_112_backup/training_history.csv';Best=91;Color=$navy},
    @{Name='Fast raw, batch 512 (early stop)';Path='results/davis/fast_run_v1/unscaled/seed_42/training_history.csv';Best=142;Color=$blue},
    @{Name='Scaled, seed 42 (stop status unknown)';Path='results/davis/fast_run_v1/scaled/seed_42/training_history.csv';Best=122;Color=$teal}
)
$c=New-Canvas 1550 820 'Davis training histories: saved training and validation MSE'
$g=$c.Graphics; $l=92; $r=1500; $t=78; $b=690; $maxEpoch=190; $maxLoss=3.6
for($k=0;$k -le 6;$k++){$v=$maxLoss*$k/6.0;$yy=$b-[int](($b-$t)*$k/6.0);$g.DrawLine($gridPen,$l,$yy,$r,$yy);$g.DrawString(('{0:0.0}' -f $v),$font,[System.Drawing.Brushes]::DimGray,35,$yy-9)}
for($k=0;$k -le 9;$k++){$v=$maxEpoch*$k/9.0;$xx=$l+[int](($r-$l)*$k/9.0);$g.DrawLine($gridPen,$xx,$t,$xx,$b);$g.DrawString(('{0:0}' -f $v),$font,[System.Drawing.Brushes]::DimGray,$xx-10,$b+8)}
$g.DrawLine($axisPen,$l,$t,$l,$b);$g.DrawLine($axisPen,$l,$b,$r,$b)
$g.DrawString('Epoch',$font,[System.Drawing.Brushes]::Black,750,760);$g.DrawString('MSE (pKd squared)',$font,[System.Drawing.Brushes]::Black,5,50)
foreach($s in $series){
    $rows=Import-Csv (Join-Path $ProjectRoot $s.Path)
    foreach($kind in @('training_mse','validation_mse')){
        $pen=New-Object System.Drawing.Pen($s.Color,2.2); if($kind -eq 'validation_mse'){$pen.DashStyle=[System.Drawing.Drawing2D.DashStyle]::Dash}
        $prev=$null
        foreach($row in $rows){$ep=[double]$row.epoch;$val=[double]$row.$kind;$px=$l+[int](($r-$l)*$ep/$maxEpoch);$py=$b-[int](($b-$t)*[Math]::Min($val,$maxLoss)/$maxLoss);$pt=New-Object System.Drawing.Point($px,$py);if($null -ne $prev){$g.DrawLine($pen,$prev,$pt)};$prev=$pt}
        $pen.Dispose()
    }
    $hist=Import-Csv (Join-Path $ProjectRoot $s.Path); $bestRow=$hist | Where-Object { [int]$_.epoch -eq $s.Best } | Select-Object -First 1
    if($bestRow){$xx=$l+[int](($r-$l)*$s.Best/$maxEpoch);$yy=$b-[int](($b-$t)*[double]$bestRow.validation_mse/$maxLoss);$br=New-Object System.Drawing.SolidBrush($s.Color);$g.FillEllipse($br,$xx-6,$yy-6,12,12);$br.Dispose()}
}
$ly=720;foreach($s in $series){$pen=New-Object System.Drawing.Pen($s.Color,3);$g.DrawLine($pen,105,$ly+7,145,$ly+7);$pen.DashStyle=[System.Drawing.Drawing2D.DashStyle]::Dash;$g.DrawLine($pen,150,$ly+7,190,$ly+7);$g.DrawString($s.Name+' (solid=train, dashed=validation; circle=best epoch '+$s.Best+')',$small,[System.Drawing.Brushes]::Black,202,$ly);$pen.Dispose();$ly+=24}
Save-Canvas $c 'training_histories.png'

$runs=@(
    @{Name='P42 raw';Json='results/davis/run_112_backup/final_metrics.json';Kind='final'},
    @{Name='F42 raw';Json='results/davis/fast_run_v1/unscaled/seed_42/final_metrics.json';Kind='final'},
    @{Name='S42 scaled';Json='results/davis/fast_run_v1/scaled/seed_42/evaluation_metrics.json';Kind='eval'},
    @{Name='Seed 44';Json='results/davis/multiseed_retry_20261016/seed_44/final_metrics.json';Kind='final'},
    @{Name='Seed 45';Json='results/davis/multiseed_retry_20261016/seed_45/final_metrics.json';Kind='final'},
    @{Name='Seed 46';Json='results/davis/multiseed_retry_20261016/seed_46/final_metrics.json';Kind='final'}
)
$payload=@{};foreach($run in $runs){$d=Get-Content -Raw (Join-Path $ProjectRoot $run.Json)|ConvertFrom-Json;if($run.Kind -eq 'eval'){$payload[$run.Name]=@([double]$d.test_mse,[double]$d.test_rmse,[double]$d.test_ci,[double]$d.test_r_m2)}else{$payload[$run.Name]=@([double]$d.test_mse,[double]$d.test_rmse,[double]$d.test_ci,[double]$d.test_r_m2)}}
$metricDefs=@(@('MSE (pKd^2)',0.5,'test MSE'),@('RMSE (pKd)',0.8,'test RMSE'),@('Concordance index',1.0,'CI'),@('Modified r_m^2',0.8,'r_m^2'))
$c=New-Canvas 1500 940 'Independent Davis test metrics by run (separate metric scales)';$g=$c.Graphics
for($m=0;$m -lt 4;$m++){$def=$metricDefs[$m];$x=25+($m%2)*740;$y=58+[Math]::Floor($m/2)*425;$vals=@();foreach($run in $runs){$vals+= [double]$payload[$run.Name][$m]};Draw-Panel -G $g -X $x -Y $y -W 710 -H 390 -Title $def[0] -Labels @('P42','F42','S42','44','45','46') -Values ([double[]]$vals) -Max ([double]$def[1]) -Unit $def[2]}
$g.DrawString('Run key: P42 = primary raw, manually stopped; F42 = fast raw; S42 = author-scaled; 44–46 = validated checkpoints. Configurations differ.',$small,[System.Drawing.Brushes]::DimGray,25,900)
Save-Canvas $c 'metric_comparison.png'

$c=New-Canvas 1150 700 'Atom-feature scaling: recorded seed-42 results';$g=$c.Graphics
$rawScale=Get-Content -Raw (Join-Path $ProjectRoot 'results/davis/fast_run_v1/unscaled/seed_42/final_metrics.json')|ConvertFrom-Json
$scaledScale=Get-Content -Raw (Join-Path $ProjectRoot 'results/davis/fast_run_v1/scaled/seed_42/evaluation_metrics.json')|ConvertFrom-Json
$scaleVals=@(@([double]$rawScale.test_mse,[double]$rawScale.test_rmse,[double]$rawScale.test_ci,[double]$rawScale.test_r_m2),@([double]$scaledScale.test_mse,[double]$scaledScale.test_rmse,[double]$scaledScale.test_ci,[double]$scaledScale.test_r_m2))
$scaleDefs=@(@('MSE (pKd^2)',.5),@('RMSE (pKd)',.8),@('CI',1.0),@('r_m^2',.8))
for($m=0;$m -lt 4;$m++){$vals=@([double]$scaleVals[0][$m],[double]$scaleVals[1][$m]);$x=20+($m%2)*560;$y=55+[Math]::Floor($m/2)*300;Draw-Panel -G $g -X $x -Y $y -W 540 -H 270 -Title $scaleDefs[$m][0] -Labels @('Raw','Scaled') -Values ([double[]]$vals) -Max ([double]$scaleDefs[$m][1]) -Unit $scaleDefs[$m][0]}
$g.DrawString('Raw: fast seed 42, batch 512, early stop at 182. Scaled: evaluation JSON; launch metadata incomplete. Not a controlled causal comparison.',$small,[System.Drawing.Brushes]::DimGray,25,650)
Save-Canvas $c 'scaling_comparison.png'

$seeds=@();foreach($sd in @(44,45,46)){$d=Get-Content -Raw (Join-Path $ProjectRoot "results/davis/multiseed_retry_20261016/seed_$sd/final_metrics.json")|ConvertFrom-Json;$seeds+=,@([double]$d.test_mse,[double]$d.test_rmse,[double]$d.test_ci,[double]$d.test_r_m2)}
$aggDefs=@(@('MSE (pKd^2)',.45),@('RMSE (pKd)',.75),@('CI',.90),@('r_m^2',.55))
$c=New-Canvas 1150 700 'Validated checkpoint metrics: seeds 44-46 (mean +/- sample SD)';$g=$c.Graphics
for($m=0;$m -lt 4;$m++){$vals=@();for($j=0;$j -lt 3;$j++){$vals+=[double]$seeds[$j][$m]};$mean=($vals|Measure-Object -Average).Average;$sum=0.0;foreach($v in $vals){$sum+=[Math]::Pow($v-$mean,2)};$sd=[Math]::Sqrt($sum/2.0);$x=20+($m%2)*560;$y=55+[Math]::Floor($m/2)*300;Draw-Panel -G $g -X $x -Y $y -W 540 -H 270 -Title $aggDefs[$m][0] -Labels @('44','45','46') -Values ([double[]]$vals) -Max ([double]$aggDefs[$m][1]) -Errors ([double[]]@($sd,$sd,$sd)) -Unit $aggDefs[$m][0];$g.DrawString(('mean {0:0.0000} +/- {1:0.0000}' -f $mean,$sd),$small,[System.Drawing.Brushes]::DimGray,$x+50,$y+28)}
$g.DrawString('Reevaluated from saved checkpoints on current test CSV; process-exit conflicts remain unresolved.',$small,[System.Drawing.Brushes]::DimGray,25,650)
Save-Canvas $c 'multiseed_variability.png'
Write-Output 'Generated report charts from saved CSV/JSON artifacts.'

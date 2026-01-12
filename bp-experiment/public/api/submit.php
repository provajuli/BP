<?php
header("Content-Type: application/json; charset=utf-8");

$raw = file_get_contents("php://input");
$data = json_decode($raw, true);

if (!$data) {
    http_response_code(400);
    echo json_encode(["ok" => false, "error" => "Invalid JSON"]);
    exit;
}

function bad($msg) {
    http_response_code(400);
    echo json_encode(["ok" => false, "error" => $msg]);
    exit;
}

$session = $data["session_id"] ?? "";
if (!preg_match('/^[a-zA-Z0-9\-]{8,64}$/', $session)) bad("Bad session_id");

$index = intval($data["index"] ?? 0);
$glyph = $data["glyph_type"] ?? "";
$sizeA = intval($data["sizeA"] ?? 0);
$sizeB = intval($data["sizeB"] ?? 0);
$sizeC = intval($data["sizeC"] ?? 0);

if ($index < 1) bad("Bad index");
if (!is_string($glyph) || $glyph === "") bad("Bad glyph_type");

foreach (["sizeA"=>$sizeA, "sizeB"=>$sizeB, "sizeC"=>$sizeC] as $k=>$v) {
    if ($v < 1 || $v > 100) bad("Bad $k");
}

$resultsDir = realpath(__DIR__ . "/../results");
if ($resultsDir === false) bad("results dir missing");

$path = $resultsDir . "/" . $session . ".csv";
$isNew = !file_exists($path);

$f = fopen($path, "a");
if (!$f) {
    http_response_code(500);
    echo json_encode(["ok" => false, "error" => "Cannot open file"]);
    exit;
}

if (function_exists("flock")) flock($f, LOCK_EX);

if ($isNew) {
    fputcsv($f, ["index","glyph_type","sizeA","sizeB","sizeC"]);
}
fputcsv($f, [$index, $glyph, $sizeA, $sizeB, $sizeC]);

if (function_exists("flock")) flock($f, LOCK_UN);
fclose($f);

echo json_encode(["ok" => true]);

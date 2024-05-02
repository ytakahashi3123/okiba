import gpxpy
import gpxpy.gpx

def write_gps_data_to_file(gps_data, output_file):
    # 新しいGPXファイルを作成
    gpx = gpxpy.gpx.GPX()

    # トラックを作成し、GPXに追加
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)

    # トラックセグメントを作成し、トラックに追加
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    # GPSデータをトラックセグメントに追加
    for point in gps_data:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=point[0], longitude=point[1], elevation=point[2]))

    # GPXファイルを出力
    with open(output_file, 'w') as f:
        f.write(gpx.to_xml())

# サンプルのGPSデータ
gps_data = [
    (35.6895, 139.6917, 0),  # 東京の緯度、経度、高度
    (40.7128, -74.0060, 10),  # ニューヨークの緯度、経度、高度
    (51.5074, -0.1278, 20),   # ロンドンの緯度、経度、高度
]

# 出力ファイル名
output_file = "gps_data.gpx"

# GPSデータをファイルに書き込む
write_gps_data_to_file(gps_data, output_file)

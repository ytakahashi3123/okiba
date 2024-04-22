import vtk as vtk

if __name__ == '__main__':
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName('surface_flow_03000.vtk')
    reader.Update()

    # vtkUnstructuredGridを取得
    input_grid = reader.GetOutput()

    # 出力用のvtkXMLUnstructuredGridWriterを作成
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('surface_flow_03000_output.vtu')

    # 出力用のvtkUnstructuredGridを作成し、readerからコピー
    output_grid = vtk.vtkUnstructuredGrid()
    output_grid.DeepCopy(input_grid)

    # WriterにvtkUnstructuredGridをセット
    writer.SetInputData(output_grid)
    writer.Write()

import React, { useState } from 'react';

function DynamicTable({ stockData }) {

  return (
    <table border="1">
      <thead>
        <tr>
          <th>Stock Name</th>
          <th>Final Value</th>
          <th>Grade</th>
          <th>Threshold</th>
          <th>Predicted Return</th>
        </tr>
      </thead>
      <tbody>
        {stockData.map(stock => (
          <tr key={stock.sheet_name}>
            <td>{stock.sheet_name}</td>
            <td>{stock.final_value}</td>
            <td>{stock.grade}</td>
            <td>{stock.optimal_threshold}</td>
            <td>{stock.predicted_return}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default DynamicTable;

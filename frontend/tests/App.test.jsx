import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../src/App.jsx';

describe('App', () => {
  it('switches between the screening and ranking views', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Screen a resume' })).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Rank candidates' }));
    expect(screen.getByRole('heading', { name: 'Rank candidates for a job' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Rank candidates' })).toHaveAttribute('aria-selected', 'true');
  });
});
